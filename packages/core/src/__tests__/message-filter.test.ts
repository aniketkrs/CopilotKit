import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Message, RunAgentInput } from "@ag-ui/client";
import { EMPTY } from "rxjs";
import { ProxiedCopilotRuntimeAgent } from "../agent";
import { CopilotKitCore } from "../core";
import * as messageFilterModule from "../core/message-filter";
import {
  ɵkeepToolResultsWithCalls,
  ɵrepairToolCallPairs,
} from "../core/message-filter";
import {
  createSuggestionsConfig,
  waitForCondition,
  createAssistantMessage,
  createMultipleToolCallsMessage,
  createMessage,
  createToolCallMessage,
  createToolResultMessage,
} from "./test-utils";

const encoder = new TextEncoder();

function createSseResponse(): Response {
  const events = [
    { type: "RUN_STARTED", threadId: "test-thread", runId: "test-run" },
    {
      type: "RUN_FINISHED",
      threadId: "test-thread",
      runId: "test-run",
      result: { newMessages: [] },
    },
  ];
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        encoder.encode(
          events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
        ),
      );
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function infoResponse(): Response {
  return new Response(
    JSON.stringify({
      version: "1.0.0",
      agents: {
        default: {
          name: "default",
          className: "HttpAgent",
          description: "assistant",
        },
      },
      audioFileTranscriptionEnabled: false,
      mode: "sse",
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

function toolCallIdsOf(message: Message): string[] {
  const calls = (message as { toolCalls?: { id: string }[] }).toolCalls ?? [];
  return calls.map((call) => call.id);
}

function toolCallIdOf(message: Message): string {
  const calls = (message as { toolCalls?: { id: string }[] }).toolCalls;
  if (!calls?.[0]) throw new Error("message carries no tool call");
  return calls[0].id;
}

/** The messages that actually went on the wire for a REST run request. */
function sentMessages(init: RequestInit): Message[] {
  return JSON.parse(init.body as string).messages;
}

describe("ɵrepairToolCallPairs", () => {
  it("restores the assistant call in front of a kept tool result", () => {
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const full = [createMessage({ id: "u1" }), call, result];

    const repaired = ɵrepairToolCallPairs([result], full);

    // Dropping the result would be valid but would throw away the only new
    // information in a human-in-the-loop turn.
    expect(repaired.map((m) => m.id)).toEqual([call.id, result.id]);
  });

  it("restores every sibling result when it restores a parallel call", () => {
    const call = createMultipleToolCallsMessage([
      { name: "lookupA" },
      { name: "lookupB" },
    ]);
    const [callA, callB] = toolCallIdsOf(call);
    const resultA = createToolResultMessage(callA!, "a");
    const resultB = createToolResultMessage(callB!, "b");
    const full = [call, resultA, resultB];

    // Keeping only the second result: a provider rejects the turn unless the
    // first call is answered too.
    const repaired = ɵrepairToolCallPairs([resultB], full);

    expect(repaired.map((m) => m.id)).toEqual([
      call.id,
      resultA.id,
      resultB.id,
    ]);
  });

  it("restores the result of a kept call, directly after the call", () => {
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const followUp = createAssistantMessage({ id: "a2" });
    const full = [createMessage({ id: "u1" }), call, result, followUp];

    const repaired = ɵrepairToolCallPairs([call, followUp], full);

    expect(repaired.map((m) => m.id)).toEqual([call.id, result.id, "a2"]);
  });

  it("leaves an open call alone when no result exists yet", () => {
    // Mid-HITL: the call is awaiting a human, so the pair is not broken.
    const call = createToolCallMessage("askUser");
    const full = [createMessage({ id: "u1" }), call];

    const repaired = ɵrepairToolCallPairs([call], full);

    expect(repaired.map((m) => m.id)).toEqual([call.id]);
  });

  it("keeps a tool result whose issuing call is not in this thread", () => {
    // No issuer to reason about, so dropping it would be the client inventing
    // a rule the backend never stated.
    const orphan = createToolResultMessage("call-from-elsewhere", "done");
    const full = [createMessage({ id: "u1" }), orphan];

    const repaired = ɵrepairToolCallPairs([orphan], full);

    expect(repaired.map((m) => m.id)).toEqual([orphan.id]);
  });

  it("preserves the filter's order and never duplicates a message", () => {
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const full = [call, result];

    const repaired = ɵrepairToolCallPairs([call, result], full);

    expect(repaired.map((m) => m.id)).toEqual([call.id, result.id]);
  });

  it("returns the list untouched when the filter kept everything", () => {
    const full = [
      createMessage({ id: "u1" }),
      createAssistantMessage({ id: "a1" }),
    ];

    expect(ɵrepairToolCallPairs(full, full)).toEqual(full);
  });
});

const userMsg = (id: string) => ({ id, role: "user", content: id }) as Message;
const textMsg = (id: string) =>
  ({ id, role: "assistant", content: id }) as Message;
const callsMsg = (id: string, ...callIds: string[]) =>
  ({
    id,
    role: "assistant",
    content: "",
    toolCalls: callIds.map((callId) => ({
      id: callId,
      type: "function",
      function: { name: "f", arguments: "{}" },
    })),
  }) as Message;
const toolMsg = (id: string, callId: string) =>
  ({ id, role: "tool", toolCallId: callId, content: id }) as Message;
const reasoningMsg = (id: string) =>
  ({ id, role: "reasoning", content: id }) as Message;
const activityMsg = (id: string) =>
  ({ id, role: "activity", activityType: "x", content: {} }) as never;

// mulberry32: a tiny seeded PRNG, so every run checks the same histories
// and a failure reproduces from its seed.
const prng = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const roleOf = (message: Message) =>
  (message as { role?: string } | null)?.role;
const slotOf = (m: Message) => (m as unknown as { slot: number }).slot;

describe("ɵkeepToolResultsWithCalls", () => {
  it("moves a detached result back behind its call (the snapshot-merge shape)", () => {
    const u = userMsg("u");
    const a1 = callsMsg("a1", "c");
    const a2 = textMsg("a2");
    const t = toolMsg("t", "c");

    expect(ɵkeepToolResultsWithCalls([u, a1, a2, t])).toEqual([u, a1, t, a2]);
  });

  it("returns the very same array for a well-formed thread", () => {
    const thread = [
      userMsg("u1"),
      callsMsg("a1", "c1", "c2"),
      toolMsg("t1", "c1"),
      activityMsg("act"),
      toolMsg("t2", "c2"),
      textMsg("a2"),
    ];

    expect(ɵkeepToolResultsWithCalls(thread)).toBe(thread);
  });

  it("leaves results in place when call ids are reused across turns", () => {
    const thread = [
      userMsg("u1"),
      callsMsg("a1", "call_0"),
      toolMsg("t1", "call_0"),
      textMsg("x1"),
      userMsg("u2"),
      callsMsg("a2", "call_0"),
      toolMsg("t2", "call_0"),
      textMsg("x2"),
    ];

    expect(ɵkeepToolResultsWithCalls(thread)).toBe(thread);
  });

  it("pairs several detached results for one reused id oldest call first", () => {
    const a1 = callsMsg("A1", "c");
    const a2 = callsMsg("A2", "c");
    const x = textMsg("X");
    const t1 = toolMsg("T1", "c");
    const t2 = toolMsg("T2", "c");

    expect(ɵkeepToolResultsWithCalls([a1, a2, x, t1, t2])).toEqual([
      a1,
      t1,
      a2,
      t2,
      x,
    ]);
  });

  it("treats one object appearing twice as two independent slots", () => {
    const a = callsMsg("a", "c");
    const t = toolMsg("t", "c");
    const x1 = textMsg("x1");
    const x2 = textMsg("x2");

    const out = ɵkeepToolResultsWithCalls([a, x1, t, a, x2, t]);

    expect(out).toHaveLength(6);
    [a, t, x1, a, t, x2].forEach((message, index) =>
      expect(out[index]).toBe(message),
    );
  });

  it("leaves a result stored before its call where it is", () => {
    const thread = [
      userMsg("u"),
      toolMsg("t", "c"),
      callsMsg("a", "c"),
      textMsg("x"),
    ];

    expect(ɵkeepToolResultsWithCalls(thread)).toBe(thread);
  });

  it("keeps both results for one call, moving only the one that answers it", () => {
    const a = callsMsg("a", "c");
    const x = textMsg("x");
    const t1 = toolMsg("t1", "c");
    const t2 = toolMsg("t2", "c");

    // t1 answers the call and moves; t2 has nothing left to answer and stays.
    expect(ɵkeepToolResultsWithCalls([a, x, t1, t2])).toEqual([a, t1, x, t2]);
  });

  it("moves a result past reasoning stored between it and its call, but not past activity", () => {
    const u = userMsg("u");
    const a = callsMsg("a", "c");
    const r = reasoningMsg("r");
    const t = toolMsg("t", "c");
    expect(ɵkeepToolResultsWithCalls([u, a, r, t])).toEqual([u, a, t, r]);

    const viaActivity = [u, a, activityMsg("act"), t];
    expect(ɵkeepToolResultsWithCalls(viaActivity)).toBe(viaActivity);
  });

  it("places a moved result after the results already following its call", () => {
    const a = callsMsg("a", "c1", "c2");
    const t1 = toolMsg("t1", "c1");
    const act = activityMsg("act");
    const x = textMsg("x");
    const t2 = toolMsg("t2", "c2");

    expect(ɵkeepToolResultsWithCalls([a, t1, act, x, t2])).toEqual([
      a,
      t1,
      act,
      t2,
      x,
    ]);
  });

  it("skips malformed entries instead of throwing, and passes them through", () => {
    const u = userMsg("u");
    const odd = {
      id: "odd",
      role: "assistant",
      toolCalls: [null, 5, { id: 3 }, { id: "c" }],
    } as unknown as Message;
    const notArray = {
      id: "na",
      role: "assistant",
      toolCalls: "nope",
    } as unknown as Message;
    const x = textMsg("x");
    const t = toolMsg("t", "c");
    const noCallId = { id: "t?", role: "tool" } as unknown as Message;
    const hole = null as unknown as Message;

    const thread = [u, odd, notArray, x, hole, noCallId, t];
    const out = ɵkeepToolResultsWithCalls(thread);

    expect(out).toEqual([u, odd, t, notArray, x, hole, noCallId]);

    const untouched = [u, hole, notArray, noCallId, x];
    expect(ɵkeepToolResultsWithCalls(untouched)).toBe(untouched);
  });

  describe("properties over random histories", () => {
    const SEED = 0x7877;
    const RUNS = 4000;
    const MAX_LENGTH = 14;
    const ID_POOL = ["c0", "c1", "c2"];

    const randomHistory = (random: () => number): Message[] => {
      const pick = <T>(items: T[]) =>
        items[Math.floor(random() * items.length)]!;
      const length = Math.floor(random() * (MAX_LENGTH + 1));
      const history: Message[] = [];
      for (let index = 0; index < length; index++) {
        // Re-insert an earlier entry (same reference) a fifth of the time.
        if (history.length > 0 && random() < 0.2) {
          history.push(pick(history));
          continue;
        }
        const id = `m${index}`;
        const kind = random();
        if (kind < 0.12) history.push(userMsg(id));
        else if (kind < 0.42) {
          const count = Math.floor(random() * 4);
          const ids = Array.from({ length: count }, () => pick(ID_POOL));
          history.push(callsMsg(id, ...ids));
        } else if (kind < 0.52) history.push(textMsg(id));
        else if (kind < 0.6) history.push(reasoningMsg(id));
        else if (kind < 0.88) history.push(toolMsg(id, pick(ID_POOL)));
        else if (kind < 0.95) history.push(activityMsg(id));
        else history.push(null as unknown as Message);
      }
      return history;
    };

    const isBoundary = (message: Message) =>
      roleOf(message) !== "tool" && roleOf(message) !== "activity";
    const counts = (messages: Message[]) => {
      const tally = new Map<Message, number>();
      for (const m of messages) tally.set(m, (tally.get(m) ?? 0) + 1);
      return tally;
    };
    /** For each position, the index of the last boundary at or before it. */
    const anchorsOf = (messages: Message[]) => {
      let anchor = -1;
      return messages.map((m, index) =>
        isBoundary(m) ? (anchor = index) : anchor,
      );
    };

    const histories = (() => {
      const random = prng(SEED);
      return Array.from({ length: RUNS }, () => randomHistory(random));
    })();

    it("never adds, drops, or duplicates a message", () => {
      for (const history of histories) {
        const out = ɵkeepToolResultsWithCalls(history);
        expect(out).toHaveLength(history.length);
        expect(counts(out)).toEqual(counts(history));
      }
    });

    it("keeps every non-tool message in its relative order", () => {
      for (const history of histories) {
        const out = ɵkeepToolResultsWithCalls(history);
        const before = history.filter((m) => roleOf(m) !== "tool");
        const after = out.filter((m) => roleOf(m) !== "tool");
        expect(after).toHaveLength(before.length);
        before.forEach((m, index) => expect(after[index]).toBe(m));
      }
    });

    it("is idempotent", () => {
      for (const history of histories) {
        const once = ɵkeepToolResultsWithCalls(history);
        const twice = ɵkeepToolResultsWithCalls(once);
        expect(twice).toBe(once);
      }
    });

    it("returns the input itself whenever nothing moved", () => {
      for (const history of histories) {
        const out = ɵkeepToolResultsWithCalls(history);
        const unchanged = out.every((m, index) => m === history[index]);
        if (unchanged) expect(out).toBe(history);
        else expect(out).not.toBe(history);
      }
    });

    it("lands every moved result behind an earlier call for its id, with only tool or activity messages between", () => {
      for (const history of histories) {
        // Distinct wrappers tag each input slot, so a moved slot can be told
        // apart from an alias of it that stayed. Position-based pairing must
        // give the same order with or without the aliasing.
        const tagged = history.map((m, slot) =>
          m === null ? m : ({ ...m, slot } as unknown as Message),
        );
        const out = ɵkeepToolResultsWithCalls(tagged);
        expect(out.map((m) => (m === null ? m : history[slotOf(m)]))).toEqual(
          ɵkeepToolResultsWithCalls(history),
        );

        const inAnchors = anchorsOf(tagged);
        const outAnchors = anchorsOf(out);
        out.forEach((m, index) => {
          if (roleOf(m) !== "tool") return;
          const slot = slotOf(m);
          const anchorBefore = inAnchors[slot]!;
          const anchorAfter = outAnchors[index]!;
          const issuerAfter = anchorAfter === -1 ? undefined : out[anchorAfter];
          if (issuerAfter === tagged[anchorBefore]) return; // stayed put
          // Moved: now behind an assistant that issued its id earlier.
          expect(roleOf(issuerAfter!)).toBe("assistant");
          expect(slotOf(issuerAfter!)).toBeLessThan(slot);
          const callIds = (
            (issuerAfter as { toolCalls?: { id: string }[] }).toolCalls ?? []
          ).map((call) => call.id);
          expect(callIds).toContain((m as { toolCallId?: string }).toolCallId);
        });
      }
    });
  });
});

describe("ProxiedCopilotRuntimeAgent messageFilter", () => {
  const originalFetch = global.fetch;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn().mockResolvedValue(createSseResponse());
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
  });

  const history = (): Message[] => [
    createMessage({ id: "u1", content: "first" }),
    createAssistantMessage({ id: "a1" }),
    createMessage({ id: "u2", content: "latest" }),
  ];

  it("sends the whole thread when no filter is set", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
    });
    agent.setMessages(history());

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
  });

  it("sends only what the filter kept", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u2"]);
  });

  it("leaves the rendered transcript at full length", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());

    await agent.runAgent();

    expect(agent.messages.map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
  });

  it("repairs the tool-call pair a blunt filter would have split", async () => {
    // The #1482 failure in reverse: trimming to the last message on a thread
    // that ends in a tool result leaves the provider a result with no call.
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages([createMessage({ id: "u1" }), call, result]);

    await agent.runAgent();

    // The call comes back, the earlier user turn stays trimmed. Sending `[]`
    // here would hand a resuming backend nothing to act on.
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([call.id, result.id]);
  });

  it("restores a dropped result so a kept call is never left unanswered", async () => {
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.filter((m) => m.role !== "tool"),
    });
    agent.setMessages([createMessage({ id: "u1" }), call, result]);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
    ]);
  });

  it("trims below run(), so the transport fork never hands a delegate a trimmed thread", async () => {
    // Placement is the Intelligence exemption. `run()` is the fork between the
    // HTTP transports and the Intelligence delegate, and the delegate forwards
    // whatever input it is handed. So the input reaching `run()` must still be
    // the whole thread; only the HTTP branch below it may narrow the payload.
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());
    const run = vi.spyOn(agent, "run");

    await agent.runAgent();

    expect(run.mock.calls[0]?.[0].messages.map((m) => m.id)).toEqual([
      "u1",
      "a1",
      "u2",
    ]);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u2"]);
  });

  it.each([
    ["auto", { transport: "auto" as const }],
    ["pending", { runtimeMode: "pending" as const }],
  ])(
    "never trims for a delegate when the mode resolves to Intelligence after the input was prepared (%s)",
    async (_label, overrides) => {
      // The regression this guards: `runAgent` prepares the input, and only
      // THEN does `run()` resolve `/info`. With the default "auto" transport,
      // or on a proxy still "pending" its first `/info`, an earlier revision
      // had already filtered by that point and handed the truncated thread to
      // the managed runtime — which is the store of record for that thread.
      fetchMock.mockImplementation(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              version: "1.0.0",
              mode: "intelligence",
              intelligence: { wsUrl: "wss://intelligence.example/ws" },
              audioFileTranscriptionEnabled: false,
              agents: { a: { name: "a", className: "HttpAgent" } },
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        ),
      );
      const agent = new ProxiedCopilotRuntimeAgent({
        runtimeUrl: "https://runtime.example",
        agentId: "a",
        messageFilter: (messages) => messages.slice(-1),
        ...overrides,
      });
      agent.setMessages(history());

      let delegateSaw: string[] | undefined;
      // Stub only the delegate construction: the mode resolution and the
      // filter — the things under test — still run for real.
      (
        agent as unknown as { resolveDelegate: () => Promise<unknown> }
      ).resolveDelegate = async () => ({
        run: (input: RunAgentInput) => {
          delegateSaw = input.messages.map((m) => m.id);
          return EMPTY;
        },
      });

      await agent.runAgent().catch(() => undefined);

      expect(delegateSaw).toEqual(["u1", "a1", "u2"]);
    },
  );

  it("falls back to the full thread when the filter smuggles in a non-message entry", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      // The shape a `m.filter(x => x.content.length > 0)` mistake produces.
      messageFilter: () => [undefined as never],
    });
    agent.setMessages(history());

    await expect(agent.runAgent()).resolves.toMatchObject({
      newMessages: expect.any(Array),
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
    // The filter produced the bad entry, so the filter is what gets blamed.
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining(
        "messageFilter returned an entry that is not a message",
      ),
    );
  });

  it("leaves Intelligence runs untrimmed, because that runtime is the store of record", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      runtimeMode: "intelligence",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());
    const run = vi.spyOn(agent, "run").mockReturnValue(EMPTY);

    await agent.runAgent().catch(() => undefined);

    expect(run.mock.calls[0]?.[0].messages.map((m) => m.id)).toEqual([
      "u1",
      "a1",
      "u2",
    ]);
  });

  it("tells the filter which agent is running", async () => {
    const seen: string[] = [];
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "billing-agent",
      transport: "rest",
      messageFilter: (messages, context) => {
        seen.push(context.agentId);
        return messages;
      },
    });
    agent.setMessages(history());

    await agent.runAgent();

    expect(seen).toEqual(["billing-agent"]);
  });

  it("hides activity messages from the filter", async () => {
    const seen: Message[][] = [];
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => {
        seen.push(messages);
        return messages;
      },
    });
    agent.setMessages([
      createMessage({ id: "u1" }),
      createMessage({ id: "act1", role: "activity" } as never),
      createMessage({ id: "u2" }),
    ]);

    await agent.runAgent();

    expect(seen[0]?.map((m) => m.id)).toEqual(["u1", "u2"]);
  });

  it("cannot corrupt the repair baseline by mutating a message it drops", async () => {
    // The filter is handed its own deep copy. Sharing the message objects with
    // `input.messages` — which is both the repair baseline and the untrimmed
    // fallback — let a filter that rewrote a dropped message's tool call hide
    // the issuer from the repair, and the kept tool result then went out
    // orphaned: a payload the provider rejects.
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => {
        const issuer = messages.find((m) => m.role === "assistant") as {
          toolCalls?: { id: string }[];
        };
        if (issuer?.toolCalls?.[0]) issuer.toolCalls[0].id = "rewritten";
        return messages.filter((m) => m.role === "tool");
      },
    });
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    agent.setMessages([createMessage({ id: "u1" }), call, result]);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const sent = sentMessages(init);
    expect(sent.map((m) => m.id)).toEqual([call.id, result.id]);
    expect(toolCallIdsOf(sent[0]!)).toEqual([toolCallIdOf(call)]);
  });

  it("cannot corrupt the transcript by mutating the array it is handed", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => {
        messages.length = 0;
        (messages as { push: (m: Message) => void }).push(
          createMessage({ id: "injected" }),
        );
        return messages;
      },
    });
    agent.setMessages(history());

    await agent.runAgent();

    expect(agent.messages.map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
  });

  it("falls back to the full thread when the filter returns a non-array", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (() => undefined) as never,
    });
    agent.setMessages(history());

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("messageFilter returned a non-array value"),
    );
  });

  it("falls back to the full thread when the filter throws", async () => {
    // Trimming is an optimization. A bug in it must not take the user's
    // message down with it.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: () => {
        throw new Error("boom");
      },
    });
    agent.setMessages(history());

    await expect(agent.runAgent()).resolves.toMatchObject({
      newMessages: expect.any(Array),
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("messageFilter failed"),
      expect.any(Error),
    );
  });

  it("blames the tool-call repair, not a filter that worked, and still sends the filter's output when the repair throws", async () => {
    // Forced through the module export: the repair failing is what is under
    // test, independent of which malformed input happens to trip it.
    vi.spyOn(messageFilterModule, "ɵrepairToolCallPairs").mockImplementation(
      () => {
        throw new Error("repair bug");
      },
    );
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());

    await expect(agent.runAgent()).resolves.toMatchObject({
      newMessages: expect.any(Array),
    });

    // The filter worked, so its trimming (or redaction) still applies; only
    // the repair on top of it is skipped.
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u2"]);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("tool-call repair failed"),
      expect.any(Error),
    );
    expect(warn).not.toHaveBeenCalledWith(
      expect.stringContaining("messageFilter failed"),
      expect.anything(),
    );
  });

  it("still sends the thread when both the filter and the repair throw", async () => {
    // A failed filter falls back to the full thread, which takes the
    // unfiltered path, so that is the step forced to throw here.
    const repair = vi.spyOn(messageFilterModule, "ɵrepairToolCallPairs");
    vi.spyOn(
      messageFilterModule,
      "ɵkeepToolResultsWithCalls",
    ).mockImplementation(() => {
      throw new Error("repair bug");
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: () => {
        throw new Error("filter bug");
      },
    });
    agent.setMessages(history());

    await expect(agent.runAgent()).resolves.toMatchObject({
      newMessages: expect.any(Array),
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
    expect(warn.mock.calls.map(([message]) => message)).toEqual([
      expect.stringContaining("messageFilter failed"),
      expect.stringContaining("tool-call repair failed"),
    ]);
    expect(repair).not.toHaveBeenCalled();
  });

  it("applies a filter assigned after construction", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
    });
    agent.setMessages(history());
    agent.messageFilter = (messages) => messages.slice(-1);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u2"]);
  });

  it("survives clone(), which the runtime does once per run", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());

    const cloned = agent.clone();
    await cloned.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u2"]);
  });

  it("trims the payload on the single-route transport too", async () => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "single",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const envelope = JSON.parse(init.body as string);
    expect(envelope.body.messages.map((m: Message) => m.id)).toEqual(["u2"]);
  });

  it("does not put back a result whose call the filter redacted from its copy", async () => {
    // The filter decides what a kept message looks like. Stripping the calls
    // from an assistant turn (and their results with it) is a valid, paired
    // payload, so the repair must not resurrect the results from the store.
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "secret");
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) =>
        messages
          .filter((m) => m.role !== "tool")
          .map((m) =>
            m.role === "assistant" ? { ...m, toolCalls: undefined } : m,
          ),
    });
    agent.setMessages([
      createMessage({ id: "u1" }),
      call,
      createAssistantMessage({ id: "a2" }),
      result,
    ]);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const sent = sentMessages(init);
    expect(sent.map((m) => m.id)).toEqual(["u1", call.id, "a2"]);
    expect(sent.some((m) => m.role === "tool")).toBe(false);
  });

  it("never sends a message twice under an identity filter, even with a shared id", async () => {
    // The filter hands back clones, so they cannot be matched to the store by
    // reference; a kept call must not also be restored from the store.
    const call = createToolCallMessage("lookup", {}, { id: "x" });
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages,
    });
    agent.setMessages([
      createMessage({ id: "u1" }),
      call,
      result,
      createAssistantMessage({ id: "x", content: "same id" }),
    ]);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const sent = sentMessages(init);
    expect(sent.length).toBeLessThanOrEqual(4);
    expect(sent.filter((m) => toolCallIdsOf(m).length > 0)).toHaveLength(1);
    expect(sent.filter((m) => m.role === "tool")).toHaveLength(1);
  });

  it("routes a filter's output through the filter repair, not the unfiltered move", async () => {
    const keep = vi.spyOn(messageFilterModule, "ɵkeepToolResultsWithCalls");
    const repair = vi.spyOn(messageFilterModule, "ɵrepairToolCallPairs");
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      messageFilter: (messages) => messages.slice(-1),
    });
    agent.setMessages(history());

    await agent.runAgent();

    expect(repair).toHaveBeenCalledTimes(1);
    expect(keep).not.toHaveBeenCalled();
  });

  it("falls back to the full thread when the filter returns a sparse array", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
      // eslint-disable-next-line no-sparse-arrays
      messageFilter: (messages) => [messages[0]!, , messages[2]!] as Message[],
    });
    agent.setMessages(history());

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u1", "a1", "u2"]);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("not a message"));
  });
});

describe("ProxiedCopilotRuntimeAgent tool-call pairing without a filter", () => {
  // The client store can hold a tool result detached from its call: when the
  // streamed TOOL_CALL_RESULT and the backend's MESSAGES_SNAPSHOT carry the
  // tool message under different ids, the snapshot merge drops the streamed
  // copy (placed after its call) and appends the snapshot's copy at the tail.
  // A backend rebuilding its state from that payload drops the result it
  // cannot pair, so every unfiltered send moves detached results back behind
  // their call (and changes nothing else).
  const originalFetch = global.fetch;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn().mockResolvedValue(createSseResponse());
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
  });

  const agentWith = (messages: Message[]) => {
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "rest",
    });
    agent.setMessages(messages);
    return agent;
  };

  /** A store whose tool result was pushed past the reply that followed it. */
  const detached = () => {
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const reply = createAssistantMessage({ id: "a2" });
    return {
      call,
      result,
      messages: [createMessage({ id: "u1" }), call, reply, result],
    };
  };

  it("puts a detached tool result back directly after its call", async () => {
    const { call, result, messages } = detached();
    const agent = agentWith(messages);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
      "a2",
    ]);
  });

  it("re-pairs on the connect request too", async () => {
    const { call, result, messages } = detached();
    const agent = agentWith(messages);

    await agent.connectAgent().catch(() => undefined);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/connect");
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
      "a2",
    ]);
  });

  it("re-pairs every result of a parallel call", async () => {
    const call = createMultipleToolCallsMessage([
      { name: "lookupA" },
      { name: "lookupB" },
    ]);
    const [callA, callB] = toolCallIdsOf(call);
    const resultA = createToolResultMessage(callA!, "a");
    const resultB = createToolResultMessage(callB!, "b");
    const agent = agentWith([
      createMessage({ id: "u1" }),
      call,
      resultB,
      createAssistantMessage({ id: "a2" }),
      resultA,
    ]);

    await agent.runAgent();

    // resultB already sits behind the call and stays; the detached resultA
    // joins it there, after the results already in place.
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      resultB.id,
      resultA.id,
      "a2",
    ]);
  });

  it("sends a well-formed thread exactly as stored", async () => {
    const call = createToolCallMessage("lookup");
    const result = createToolResultMessage(toolCallIdOf(call), "done");
    const messages = [
      createMessage({ id: "u1" }),
      call,
      result,
      createAssistantMessage({ id: "a2" }),
      createMessage({ id: "u2" }),
    ];
    const agent = agentWith(messages);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init)).toEqual(messages);
  });

  it("leaves an open call alone", async () => {
    const call = createToolCallMessage("approve");
    const agent = agentWith([createMessage({ id: "u1" }), call]);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual(["u1", call.id]);
  });

  it("still re-pairs the full thread a broken filter falls back to", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const { call, result, messages } = detached();
    const agent = agentWith(messages);
    agent.messageFilter = () => {
      throw new Error("boom");
    };

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
      "a2",
    ]);
  });

  it("still re-pairs the full thread a non-array filter result falls back to", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const { call, result, messages } = detached();
    const agent = agentWith(messages);
    agent.messageFilter = (() => "not an array") as never;

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
      "a2",
    ]);
  });

  it("re-pairs on the single-route connect request too", async () => {
    const { call, result, messages } = detached();
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "single",
    });
    agent.setMessages(messages);

    await agent.connectAgent().catch(() => undefined);

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const envelope = JSON.parse(init.body as string);
    expect(envelope.method).toBe("agent/connect");
    expect(envelope.body.messages.map((m: Message) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
      "a2",
    ]);
  });

  it("re-pairs on the single-route run request too", async () => {
    const { call, result, messages } = detached();
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "single",
    });
    agent.setMessages(messages);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const envelope = JSON.parse(init.body as string);
    expect(envelope.method).toBe("agent/run");
    expect(envelope.body.messages.map((m: Message) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
      "a2",
    ]);
  });

  it("sends every stored message, even two that share an id", async () => {
    // The unfiltered step works by position, so a duplicated id (a re-seeded
    // store, a backend that reuses ids) is never mistaken for a repeat.
    const { call, result, messages } = detached();
    const dupe = createMessage({ id: "u1", content: "again" });
    const agent = agentWith([...messages, dupe]);

    await agent.runAgent();

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      result.id,
      "a2",
      "u1",
    ]);
  });

  it("sends a malformed stored thread as-is instead of failing the run", async () => {
    // A persisted or hand-built message can carry a null entry in its
    // toolCalls. The unfiltered pairing step skips entries it cannot read
    // rather than throwing; it must never cost the user a send the agent made
    // before it existed. The single-route transport is used because it
    // forwards the payload without schema validation, so before the pairing
    // step existed this thread did go out.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const malformed = createAssistantMessage({
      id: "a1",
      toolCalls: [null] as never,
    });
    const agent = new ProxiedCopilotRuntimeAgent({
      runtimeUrl: "https://runtime.example",
      agentId: "a",
      transport: "single",
    });
    agent.setMessages([
      createMessage({ id: "u1" }),
      malformed,
      createMessage({ id: "u2" }),
    ]);

    await expect(agent.runAgent()).resolves.toMatchObject({
      newMessages: expect.any(Array),
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const envelope = JSON.parse(init.body as string);
    expect(envelope.body.messages.map((m: Message) => m.id)).toEqual([
      "u1",
      "a1",
      "u2",
    ]);
    expect(warn).not.toHaveBeenCalled();
  });

  it("warns once, naming the repair, and sends the input unchanged when the repair throws", async () => {
    // Forced through the module export: no known input makes it throw.
    vi.spyOn(
      messageFilterModule,
      "ɵkeepToolResultsWithCalls",
    ).mockImplementation(() => {
      throw new Error("repair bug");
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { call, result, messages } = detached();
    const agent = agentWith(messages);

    await expect(agent.runAgent()).resolves.toMatchObject({
      newMessages: expect.any(Array),
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(sentMessages(init).map((m) => m.id)).toEqual([
      "u1",
      call.id,
      "a2",
      result.id,
    ]);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("tool-call repair failed"),
      expect.any(Error),
    );
  });
});

describe("CopilotKitCore messageFilter reaches runtime-discovered agents", () => {
  const originalFetch = global.fetch;
  const originalWindow = (global as { window?: unknown }).window;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi
      .fn()
      .mockImplementation((url: string) =>
        Promise.resolve(
          String(url).includes("/info") ? infoResponse() : createSseResponse(),
        ),
      );
    global.fetch = fetchMock as unknown as typeof fetch;
    // The core only opens its `/info` connection in a browser.
    (global as { window?: unknown }).window = {};
  });

  afterEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete (global as { window?: unknown }).window;
    } else {
      (global as { window?: unknown }).window = originalWindow;
    }
  });

  const runRequestBody = (): Message[] => {
    // `endsWith`, not `includes`: the `/info` URL contains "/runtime.example".
    const call = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/run"),
    ) as [string, RequestInit];
    return sentMessages(call[1]);
  };

  it("trims the payload of an agent the runtime advertised", async () => {
    // The case #1482 reporters actually have: no agent instance of their own,
    // just `<CopilotKit runtimeUrl>` and an agent discovered from `/info`.
    const core = new CopilotKitCore({
      runtimeUrl: "https://runtime.example",
      messageFilter: (messages) => messages.slice(-1),
    });
    await waitForCondition(() => Boolean(core.getAgent("default")));

    const agent = core.getAgent("default")!;
    agent.setMessages([
      createMessage({ id: "u1" }),
      createAssistantMessage({ id: "a1" }),
      createMessage({ id: "u2" }),
    ]);
    await agent.runAgent();

    expect(runRequestBody().map((m) => m.id)).toEqual(["u2"]);
  });

  it("applies a filter set after the agent was discovered", async () => {
    const core = new CopilotKitCore({ runtimeUrl: "https://runtime.example" });
    await waitForCondition(() => Boolean(core.getAgent("default")));

    core.setMessageFilter((messages) => messages.slice(-1));

    const agent = core.getAgent("default")!;
    agent.setMessages([
      createMessage({ id: "u1" }),
      createMessage({ id: "u2" }),
    ]);
    await agent.runAgent();

    expect(runRequestBody().map((m) => m.id)).toEqual(["u2"]);
  });

  it("goes back to the full thread when the filter is cleared", async () => {
    const core = new CopilotKitCore({
      runtimeUrl: "https://runtime.example",
      messageFilter: (messages) => messages.slice(-1),
    });
    await waitForCondition(() => Boolean(core.getAgent("default")));

    core.setMessageFilter(undefined);

    const agent = core.getAgent("default")!;
    agent.setMessages([
      createMessage({ id: "u1" }),
      createMessage({ id: "u2" }),
    ]);
    await agent.runAgent();

    expect(runRequestBody().map((m) => m.id)).toEqual(["u1", "u2"]);
  });
});

describe("messageFilter and suggestions", () => {
  const originalFetch = global.fetch;
  const originalWindow = (global as { window?: unknown }).window;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // No `suggestions` capability advertised, so the engine takes the
    // clone + runAgent fallback rather than the stateless /suggest path.
    fetchMock = vi.fn().mockImplementation((url: string) =>
      Promise.resolve(
        String(url).includes("/info")
          ? new Response(
              JSON.stringify({
                version: "1.0.0",
                mode: "sse",
                audioFileTranscriptionEnabled: false,
                agents: {
                  default: { name: "default", className: "HttpAgent" },
                  consumer: { name: "consumer", className: "HttpAgent" },
                },
              }),
              { status: 200, headers: { "content-type": "application/json" } },
            )
          : createSseResponse(),
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;
    (global as { window?: unknown }).window = {};
  });

  afterEach(() => {
    vi.restoreAllMocks();
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete (global as { window?: unknown }).window;
    } else {
      (global as { window?: unknown }).window = originalWindow;
    }
  });

  it("keeps the seeded context on a suggestion run", async () => {
    // A suggestion clone runs under a fresh thread id, so no backend store can
    // fill in what a filter withheld: the seeded messages ARE the context. If
    // the app's filter rode the clone, the provider would be asked to suggest
    // from a single message.
    const core = new CopilotKitCore({
      runtimeUrl: "https://runtime.example",
      messageFilter: (messages) => messages.slice(-1),
    });
    await waitForCondition(() => Boolean(core.getAgent("consumer")));

    core
      .getAgent("consumer")!
      .setMessages([
        createMessage({ id: "c1", content: "first" }),
        createAssistantMessage({ id: "c2" }),
        createMessage({ id: "c3", content: "latest" }),
      ]);
    core.addSuggestionsConfig(
      createSuggestionsConfig({
        providerAgentId: "default",
        consumerAgentId: "consumer",
      }),
    );

    core.reloadSuggestions("consumer");

    await vi.waitFor(() => {
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith("/agent/default/run"),
        ),
      ).toBe(true);
    });

    const call = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/agent/default/run"),
    ) as [string, RequestInit];
    // Trimmed, this would be exactly one message.
    expect(sentMessages(call[1]).length).toBeGreaterThan(1);
  });
});
