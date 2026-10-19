import type { Message } from "@ag-ui/client";

/**
 * Rewrites the message list a runtime agent sends on each run.
 *
 * CopilotKit sends the whole thread on every request. When the backend already
 * stores the conversation (a LangGraph checkpointer, a Mastra memory store, a
 * Strands `SessionManager`, an Agent Framework chat history provider), that
 * history is dead weight: the payload grows without bound, and an agent that
 * merges the inbound list with its own store can show the model every turn
 * twice. See https://github.com/CopilotKit/CopilotKit/issues/1482.
 *
 * The filter runs on the outbound payload only. It never touches the messages
 * the UI renders, so trimming here does not shorten the visible transcript.
 *
 * Return the messages to send. Returning an empty array is valid and means
 * "send nothing but the run metadata".
 *
 * @param messages - The thread as the client holds it, in order, with
 *   `activity`-role messages already removed.
 * @param context - Which agent is about to run.
 */
export type CopilotKitMessageFilter = (
  messages: Message[],
  context: { agentId: string },
) => Message[];

function isMessage(entry: unknown): entry is Message {
  return typeof entry === "object" && entry !== null;
}

/**
 * The call ids an assistant turn issues, in order. A malformed `toolCalls`
 * (not an array, null entries, non-string ids) contributes nothing rather
 * than throwing. The empty string is a valid id.
 */
function toolCallIdsOf(message: Message): string[] {
  if (message.role !== "assistant") return [];
  const calls: unknown = (message as { toolCalls?: unknown }).toolCalls;
  if (!Array.isArray(calls)) return [];
  const ids: string[] = [];
  for (const call of calls) {
    const id: unknown = isMessage(call)
      ? (call as { id?: unknown }).id
      : undefined;
    if (typeof id === "string") ids.push(id);
  }
  return ids;
}

function resultCallIdOf(message: Message): string | undefined {
  if (message.role !== "tool") return undefined;
  const id: unknown = (message as { toolCallId?: unknown }).toolCallId;
  return typeof id === "string" ? id : undefined;
}

/**
 * Pairs each tool result in `thread` with the assistant turn that issued it,
 * positionally: a result belongs to the nearest assistant turn before it that
 * issued its call id. Ids alone cannot do this — some backends number calls
 * per turn (`call_0` every turn), and one call can be answered twice. A result
 * with no issuer before it stays unpaired.
 *
 * Also settles the order each issuer's results go out in. Results already
 * sitting directly behind their issuer keep their stored order, so a
 * well-formed turn is never reshuffled. If any was detached, they follow the
 * order the turn issued its calls in, and results answering the same call
 * keep their stored order.
 */
function pairByPosition(thread: Message[]) {
  const issuerOf = new Map<Message, Message>();
  const resultsOf = new Map<Message, Message[]>();
  const positionOf = new Map<Message, number>();
  const latestIssuer = new Map<string, Message>();

  thread.forEach((message, position) => {
    if (!isMessage(message)) return;
    const resultCallId = resultCallIdOf(message);
    if (resultCallId !== undefined) {
      const issuer = latestIssuer.get(resultCallId);
      if (issuer) {
        issuerOf.set(message, issuer);
        resultsOf.get(issuer)!.push(message);
      }
      return;
    }
    const callIds = toolCallIdsOf(message);
    if (callIds.length === 0) return;
    resultsOf.set(message, []);
    positionOf.set(message, position);
    for (const callId of callIds) latestIssuer.set(callId, message);
  });

  for (const [issuer, results] of resultsOf) {
    const start = positionOf.get(issuer)! + 1;
    const alreadyBehindIssuer = results.every(
      (result, offset) => thread[start + offset] === result,
    );
    if (alreadyBehindIssuer) continue;
    const callIds = toolCallIdsOf(issuer);
    const rank = (result: Message) => callIds.indexOf(resultCallIdOf(result)!);
    // Array.prototype.sort is stable, so results answering one call keep
    // their stored order.
    results.sort((a, b) => rank(a) - rank(b));
  }

  return { issuerOf, resultsOf };
}

/**
 * Maps each entry of `kept` to the message in `full` it stands for. Without a
 * filter they are the same objects. A filter is handed a deep clone, so its
 * entries are matched by id — but only when that id names exactly one message
 * in `full`. An entry with an unknown or ambiguous id maps to itself, which
 * means it is passed through without repair.
 */
function matchKeptToFull(kept: Message[], full: Message[]) {
  const stored = new Set<Message>(full.filter(isMessage));
  const byId = new Map<unknown, Message | null>();
  for (const message of stored) {
    byId.set(message.id, byId.has(message.id) ? null : message);
  }
  const storedFor = new Map<Message, Message>();
  for (const entry of kept) {
    if (!isMessage(entry) || storedFor.has(entry)) continue;
    const match = stored.has(entry)
      ? entry
      : entry.id === undefined
        ? undefined
        : byId.get(entry.id);
    storedFor.set(entry, match ?? entry);
  }
  return storedFor;
}

/**
 * Repair the tool-call pairs a filter broke, or that the client store holds
 * out of order.
 *
 * It runs on every send, filter or not, with `kept === full` when there is no
 * filter, so it has to be safe on any history the store can hold. It only
 * re-places or restores messages: it never drops one, never emits one twice,
 * and a well-formed thread goes out unchanged. The store can detach a tool
 * result from its call on its own: when a backend's `MESSAGES_SNAPSHOT`
 * carries a tool message under a different id than the streamed
 * `TOOL_CALL_RESULT`, the `@ag-ui/client` merge drops the copy placed after its
 * call and appends the snapshot's copy at the end. Each assistant turn here
 * pulls its results in directly behind it, so that order goes out repaired.
 *
 * Every provider rejects a half pair. A tool result whose call is gone reads as
 * a reply to nothing, and an assistant tool call whose result is gone leaves
 * the turn unanswered — the failure reporters hit on #1482 when a trimmed
 * history landed mid-HITL and Anthropic answered "a tool result is expected".
 * A filter written as `messages.slice(-1)` cannot know this, so the repair runs
 * for every filter rather than being something each caller opts into.
 *
 * Pairs are read from the untrimmed thread, positionally: a result belongs to
 * the nearest assistant turn before it in `full` that issued its call id. No
 * id is treated as a unique key — a call id can recur every turn, one call can
 * have two results, and two messages can share a message id (or have none) —
 * so the only thing suppressed is the repair re-emitting a message it already
 * placed, judged by object identity.
 *
 * Both repairs restore, never discard:
 *
 * - A kept tool result whose assistant call was trimmed away gets that call put
 *   back in front of it. Dropping the result instead would be protocol-safe and
 *   semantically wrong: on a human-in-the-loop turn the result IS the new
 *   information, and a backend that already holds the call would be asked to
 *   resume with nothing.
 * - A kept assistant call gets every result paired with it placed directly
 *   after it, whether the filter trimmed them or the store detached them.
 *   Results already directly behind their call keep their order; otherwise
 *   they follow the order the turn issued its calls in.
 *
 * Restoring a call pulls in every result for that call, not only the one the
 * filter kept. Providers reject an assistant turn with a parallel tool call
 * left unanswered, so a half-answered set is as invalid as a missing one.
 *
 * Left alone:
 *
 * - A call with no result after it in `full`. That is an open call (a pending
 *   frontend tool, an unanswered interrupt), not a broken pair.
 * - A result with no issuer before it in `full`: its call is missing, or is
 *   stored after it. It stays where it is. Hoisting a later issuer in front of
 *   it would detach that turn from the reasoning stored before it.
 * - Assistant turns. One moves only to restore a call the filter trimmed away,
 *   so an unanswered turn reusing an answered call's id stays put.
 * - Messages the filter synthesized (absent from `full`, or carrying an id
 *   that is not unique there). They pass through untouched, and the filter's
 *   own ordering is otherwise preserved. A kept message goes out as the
 *   filter's copy, not the stored original.
 *
 * Malformed input does not throw: non-message entries of `full`, a `toolCalls`
 * that is not an array, null calls and non-string ids are skipped. A
 * non-message entry the filter itself added (one that is not in `full`) does
 * throw, so the caller falls back to the untrimmed thread.
 */
export function ɵrepairToolCallPairs(
  kept: Message[],
  full: Message[],
): Message[] {
  const { issuerOf, resultsOf } = pairByPosition(full);
  const storedFor = matchKeptToFull(kept, full);
  // The filter's copy of each stored message it kept, so a message the repair
  // places early still goes out as the filter left it.
  const keptCopyOf = new Map<Message, Message>();
  for (const [entry, stored] of storedFor) {
    if (!keptCopyOf.has(stored)) keptCopyOf.set(stored, entry);
  }

  const repaired: Message[] = [];
  // Keyed by the stored message, so a filter's copy and its original count as
  // one message.
  const emitted = new Set<Message>();
  // Messages placed ahead of their own slot in `kept`; the loop skips that
  // slot when it reaches it.
  const placedAhead = new Set<Message>();

  const pull = (stored: Message) => {
    if (emitted.has(stored)) return;
    emitted.add(stored);
    placedAhead.add(stored);
    repaired.push(keptCopyOf.get(stored) ?? stored);
  };
  const pullResultsOf = (issuer: Message) => {
    for (const result of resultsOf.get(issuer) ?? []) pull(result);
  };

  for (const entry of kept) {
    if (!isMessage(entry)) {
      if (full.includes(entry)) continue;
      throw new TypeError(
        "ɵrepairToolCallPairs: the filtered list holds an entry that is not a message",
      );
    }
    const stored = storedFor.get(entry)!;
    if (placedAhead.delete(stored)) continue;

    const issuer = issuerOf.get(stored);
    if (issuer && !emitted.has(issuer)) {
      // Its call was trimmed away (or the filter moved it later): put the call
      // back in front of this result, followed by all of its results.
      pull(issuer);
      pullResultsOf(issuer);
      if (placedAhead.delete(stored)) continue;
    }

    emitted.add(stored);
    repaired.push(entry);
    pullResultsOf(stored);
  }

  return repaired;
}
