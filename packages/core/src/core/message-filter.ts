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

interface AssistantToolCall {
  id: string;
}

function toolCallIdsOf(message: Message): string[] {
  const calls = (message as { toolCalls?: AssistantToolCall[] }).toolCalls;
  if (!Array.isArray(calls)) return [];
  return calls.map((call) => call.id).filter((id): id is string => Boolean(id));
}

function resultCallIdOf(message: Message): string | undefined {
  if (message.role !== "tool") return undefined;
  return (message as { toolCallId?: string }).toolCallId;
}

/**
 * Repair the tool-call pairs a filter broke.
 *
 * Every provider rejects a half pair. A tool result whose call is gone reads as
 * a reply to nothing, and an assistant tool call whose result is gone leaves
 * the turn unanswered — the failure reporters hit on #1482 when a trimmed
 * history landed mid-HITL and Anthropic answered "a tool result is expected".
 * A filter written as `messages.slice(-1)` cannot know this, so the repair runs
 * for every filter rather than being something each caller opts into.
 *
 * Both repairs restore, never discard, and both are anchored on the untrimmed
 * thread:
 *
 * - A kept tool result whose assistant call was trimmed away gets that call put
 *   back in front of it. Dropping the result instead would be protocol-safe and
 *   semantically wrong: on a human-in-the-loop turn the result IS the new
 *   information, and a backend that already holds the call would be asked to
 *   resume with nothing.
 * - A kept assistant call whose result was trimmed gets that result restored,
 *   placed directly after the call.
 *
 * Restoring a call pulls in every result for that call, not only the one the
 * filter kept. Providers reject an assistant turn with a parallel tool call
 * left unanswered, so a half-answered set is as invalid as a missing one.
 *
 * A call with no result anywhere in `full` is left alone. That is an open call
 * (a pending frontend tool, an unanswered interrupt), not a broken pair. A
 * result whose call is nowhere in `full` is left alone too: it was already
 * unpaired before the filter ran, so it is not this function's to fix.
 *
 * The filter's own ordering is preserved. Restorations are the only insertions,
 * and messages the filter synthesized (absent from `full`) pass through
 * untouched.
 */
export function ɵrepairToolCallPairs(
  kept: Message[],
  full: Message[],
): Message[] {
  // Both maps are built from the untrimmed thread: the repair has to reason
  // about pairs the filter already broke, so the kept list cannot answer
  // "did this call have a result".
  const callIdToIssuer = new Map<string, Message>();
  const callIdToResult = new Map<string, Message>();
  for (const message of full) {
    for (const callId of toolCallIdsOf(message)) {
      callIdToIssuer.set(callId, message);
    }
    const resultCallId = resultCallIdOf(message);
    if (resultCallId !== undefined) {
      callIdToResult.set(resultCallId, message);
    }
  }

  const repaired: Message[] = [];
  const emitted = new Set<string>();

  const emit = (message: Message) => {
    if (emitted.has(message.id)) return;
    emitted.add(message.id);
    repaired.push(message);
  };

  /** Emit an assistant turn together with every result it is owed. */
  const emitWithResults = (message: Message) => {
    emit(message);
    for (const callId of toolCallIdsOf(message)) {
      const result = callIdToResult.get(callId);
      if (result) emit(result);
    }
  };

  for (const message of kept) {
    const resultCallId = resultCallIdOf(message);
    if (resultCallId === undefined) {
      emitWithResults(message);
      continue;
    }

    const issuer = callIdToIssuer.get(resultCallId);
    if (issuer) {
      // Emits the issuer first, then this result along with its siblings.
      emitWithResults(issuer);
      continue;
    }
    emit(message);
  }

  return repaired;
}

/**
 * Move each detached tool result back behind the assistant call that issued
 * it, and change nothing else. Used for unfiltered sends only; a configured
 * filter goes through {@link ɵrepairToolCallPairs} instead.
 *
 * The client store can detach a tool result from its call on its own: when a
 * backend's `MESSAGES_SNAPSHOT` carries a tool message under a different id
 * than the streamed `TOOL_CALL_RESULT`, the `@ag-ui/client` merge drops the
 * copy placed after its call and appends the snapshot's copy at the end,
 * leaving `assistant(tool_call), assistant(text), tool(result)`. A backend that
 * rebuilds its state from that payload discards every result it cannot pair.
 *
 * The move is deliberately minimal, and it is the same algorithm the
 * `@ag-ui/client` snapshot reducer applies:
 *
 * - Everything is by list position. The same object or id appearing twice is
 *   two independent slots. Nothing is dropped, deduplicated, or added, and
 *   only tool-role messages ever change position. Duplicated entries
 *   (including an assistant call the merge duplicated) are never dropped or
 *   deduplicated either, so a duplicated call can be left unanswered.
 * - The anchor is the last message so far that is neither `tool` nor
 *   `activity`. Each tool call an assistant issues opens a slot for its call
 *   id.
 * - A result answers the nearest earlier unanswered call with its id (the
 *   newest open slot). If that slot was issued at the anchor, the result is
 *   already in place and stays put. Otherwise it moves behind that issuer,
 *   after the tool and activity messages already following it, in stored
 *   order. A result with no open slot (its call is later, absent, or already
 *   answered) stays put.
 * - Nearest, not oldest, because providers reuse ids like `call_0` every
 *   turn: an earlier call that was never answered must not pull a later
 *   turn's result back across user turns.
 * - Any other message between a call and its result separates them, so the
 *   result moves before it: reasoning stored between the two ends up after
 *   the result. Activity messages do not separate a result from its call.
 * - Malformed entries (non-object messages, non-array `toolCalls`, call
 *   entries without a string `id`) are ignored rather than thrown on.
 *
 * Returns `messages` itself when nothing moved.
 */
export function ɵkeepToolResultsWithCalls(messages: Message[]): Message[] {
  const openSlots = new Map<string, { issuer: number; answered: boolean }[]>();
  const movedBehind = new Map<number, number[]>();
  const moved = new Set<number>();
  let anchor = -1;

  for (let index = 0; index < messages.length; index++) {
    const message: unknown = messages[index];
    const role = roleOf(message);

    if (role === "activity") continue;

    if (role !== "tool") {
      anchor = index;
      const calls =
        role === "assistant"
          ? (message as { toolCalls?: unknown }).toolCalls
          : undefined;
      if (!Array.isArray(calls)) continue;
      for (const call of calls) {
        if (typeof call !== "object" || call === null) continue;
        const callId = (call as { id?: unknown }).id;
        if (typeof callId !== "string") continue;
        const slots = openSlots.get(callId) ?? [];
        slots.push({ issuer: index, answered: false });
        openSlots.set(callId, slots);
      }
      continue;
    }

    const callId = (message as { toolCallId?: unknown }).toolCallId;
    if (typeof callId !== "string") continue;
    const slots = openSlots.get(callId);
    if (!slots) continue;

    let nearest: { issuer: number; answered: boolean } | undefined;
    for (let slot = slots.length - 1; slot >= 0 && !nearest; slot--) {
      if (!slots[slot]!.answered) nearest = slots[slot];
    }
    if (!nearest) continue;
    nearest.answered = true;
    // A slot issued at the anchor is the newest one there can be, so the
    // result is already where it belongs.
    if (nearest.issuer === anchor) continue;
    moved.add(index);
    const behind = movedBehind.get(nearest.issuer) ?? [];
    behind.push(index);
    movedBehind.set(nearest.issuer, behind);
  }

  if (moved.size === 0) return messages;

  const result: Message[] = [];
  let openIssuer = -1;
  const flush = () => {
    for (const index of movedBehind.get(openIssuer) ?? []) {
      result.push(messages[index] as Message);
    }
  };
  for (let index = 0; index < messages.length; index++) {
    if (moved.has(index)) continue;
    const role = roleOf(messages[index]);
    if (role !== "tool" && role !== "activity") {
      flush();
      openIssuer = index;
    }
    result.push(messages[index] as Message);
  }
  flush();
  return result;
}

function roleOf(message: unknown): unknown {
  if (typeof message !== "object" || message === null) return undefined;
  return (message as { role?: unknown }).role;
}
