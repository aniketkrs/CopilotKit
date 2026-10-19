import type { ReactNode } from "react";
import { act, fireEvent, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { createGenerativeUI } from "../catalog";
import { createGenerativeUIDispatcher } from "../dispatcher";
import { renderSurface } from "../standard/__tests__/render";

function Text({ text }: { text: string | number }) {
  return <span data-testid="text">{String(text)}</span>;
}
function Rows({ rows }: { rows: Array<{ label: string; value: unknown }> }) {
  return (
    <dl>
      {rows.map((row) => (
        <div key={row.label} data-testid={`row-${row.label}`}>
          {row.label}: {String(row.value)}
        </div>
      ))}
    </dl>
  );
}
function Button({
  label,
  onClick,
}: {
  label: string;
  onClick?: (args: Record<string, never>) => Promise<unknown>;
}) {
  return (
    <button type="button" onClick={() => void onClick?.({})}>
      {label}
    </button>
  );
}
function Column({ children }: { children?: ReactNode }) {
  return <div>{children}</div>;
}

const ui = createGenerativeUI({
  id: "test",
  components: {
    Column: { component: Column, slots: { children: "multiple" } },
    Text: {
      component: Text,
      props: z.object({ text: z.union([z.string(), z.number()]) }),
    },
    Rows: {
      component: Rows,
      props: z.object({
        rows: z.array(z.object({ label: z.string(), value: z.unknown() })),
      }),
    },
    Button: {
      component: Button,
      props: z.object({ label: z.string() }),
      events: { onClick: { parameters: z.object({}) } },
    },
  },
});

const setData = (path: string, value: unknown) => ({
  version: "v1.0" as const,
  updateDataModel: { surfaceId: "surface", path, value },
});

describe("dynamic values in props", () => {
  it("resolves a binding nested in a list prop and updates when the data changes", async () => {
    const v = renderSurface(
      ui,
      [
        {
          id: "root",
          component: "Rows",
          rows: [
            { label: "Price", value: { "@path": "/price" } },
            { label: "Fixed", value: 1 },
          ],
        },
      ],
      { price: 10 },
    );
    expect((await v.findByTestId("row-Price")).textContent).toBe("Price: 10");
    v.update(setData("/price", 12));
    await waitFor(() =>
      expect(v.getByTestId("row-Price").textContent).toBe("Price: 12"),
    );
    expect(v.reportError).not.toHaveBeenCalled();
  });

  it("computes a function value from bound data and recomputes on change", async () => {
    const v = renderSurface(
      ui,
      [
        {
          id: "root",
          component: "Text",
          text: {
            "@call": "formatString",
            args: {
              value:
                "${/qty} × ${formatCurrency(value: /price, currency: 'EUR')}",
            },
          },
        },
      ],
      { qty: 2, price: 1599 },
    );
    expect((await v.findByTestId("text")).textContent).toBe("2 × €1,599.00");
    v.update(setData("/qty", 3));
    await waitFor(() =>
      expect(v.getByTestId("text").textContent).toBe("3 × €1,599.00"),
    );
  });

  it("falls back and reports when a function value cannot be computed", async () => {
    const v = renderSurface(ui, [
      {
        id: "root",
        component: "Text",
        text: { "@call": "formatNumber", args: { value: "many" } },
      },
    ]);
    await waitFor(() => expect(v.reportError).toHaveBeenCalled());
    const error = v.reportError.mock.calls[0][0];
    expect(error.kind).toBe("props");
    expect(error.error.message).toContain("text: value must be a number");
    expect(v.queryByTestId("text")).toBeNull();
  });

  it("recomputes a series when the data it reads changes", async () => {
    const v = renderSurface(
      ui,
      [
        {
          id: "root",
          component: "Rows",
          rows: {
            "@call": "series",
            args: {
              count: 2,
              variable: "m",
              item: { label: "M${m}", value: "${m * seats * 29}" },
              seats: { "@path": "/seats" },
            },
          },
        },
      ],
      { seats: 2 },
    );
    expect((await v.findByTestId("row-M2")).textContent).toBe("M2: 116");
    v.update(setData("/seats", 3));
    await waitFor(() =>
      expect(v.getByTestId("row-M2").textContent).toBe("M2: 174"),
    );
  });

  it("gives @index to a template item inside a function value", async () => {
    const v = renderSurface(
      ui,
      [
        {
          id: "root",
          component: "Column",
          children: { componentId: "row", path: "/items" },
        },
        {
          id: "row",
          component: "Text",
          text: {
            "@call": "formatString",
            args: { value: "${@index(offset: 1)}. ${label}" },
          },
        },
      ],
      { items: [{ label: "a" }, { label: "b" }] },
    );
    const texts = await v.findAllByTestId("text");
    expect(texts.map((t) => t.textContent)).toEqual(["1. a", "2. b"]);
  });
});

describe("dynamic values in event bindings", () => {
  afterEach(() => vi.restoreAllMocks());

  it("opens a URL from a functionCall binding, as a user gesture", async () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    const v = renderSurface(ui, [
      {
        id: "root",
        component: "Button",
        label: "Docs",
        onClick: {
          functionCall: {
            "@call": "openUrl",
            args: { url: "https://example.com/docs" },
          },
        },
      },
    ]);
    fireEvent.click(await v.findByRole("button", { name: "Docs" }));
    await waitFor(() =>
      expect(open).toHaveBeenCalledWith(
        "https://example.com/docs",
        "_blank",
        "noopener,noreferrer",
      ),
    );
    expect(v.forwardEvent).not.toHaveBeenCalled();
    expect(v.reportError).not.toHaveBeenCalled();
  });

  it("resolves function values in the userMessage and the context of an event", async () => {
    const v = renderSurface(
      ui,
      [
        {
          id: "root",
          component: "Button",
          label: "Buy",
          onClick: {
            event: {
              name: "buy",
              userMessage: {
                "@call": "formatString",
                args: { value: "Buy ${/qty} for ${/name}" },
              },
              context: {
                total: {
                  "@call": "formatCurrency",
                  args: { value: { "@path": "/price" }, currency: "USD" },
                },
                nested: { qty: { "@path": "/qty" } },
              },
            },
          },
        },
      ],
      { qty: 2, name: "Ada", price: 5 },
    );
    fireEvent.click(await v.findByRole("button", { name: "Buy" }));
    await waitFor(() => expect(v.forwardEvent).toHaveBeenCalled());
    const [event] = v.forwardEvent.mock.calls[0];
    expect(event.userMessage).toBe("Buy 2 for Ada");
    expect(event.context).toEqual({
      total: "$5.00",
      nested: { qty: 2 },
    });
  });
});

describe("basic functions through the dispatcher", () => {
  const dispatcher = createGenerativeUIDispatcher({
    getDeclared: () => ui.functions,
  });

  it("runs a basic function for the agent, except one that needs a user gesture", async () => {
    await expect(
      dispatcher.invoke(
        "formatNumber",
        { value: 1234 },
        { functionCallId: "f1" },
      ),
    ).resolves.toEqual({ ok: true, result: "1,234" });
    const refused = await dispatcher.invoke(
      "openUrl",
      { url: "https://x" },
      { functionCallId: "f2" },
    );
    expect(refused.ok).toBe(false);
    if (!refused.ok) {
      expect(refused.error.code).toBe("invalid_function_call");
    }
  });

  it("returns invalid_arguments for bad args and function_not_found for other names", async () => {
    const bad = await dispatcher.invoke(
      "formatCurrency",
      { value: 1 },
      { functionCallId: "f3" },
    );
    expect(bad.ok).toBe(false);
    if (!bad.ok) expect(bad.error.code).toBe("invalid_arguments");
    const missing = await dispatcher.invoke(
      "nope",
      {},
      { functionCallId: "f4" },
    );
    expect(missing.ok).toBe(false);
    if (!missing.ok) expect(missing.error.code).toBe("function_not_found");
  });
});

describe("ingest", () => {
  it("does not reject a prop with a nested binding at ingest", () => {
    const v = renderSurface(
      ui,
      [
        {
          id: "root",
          component: "Rows",
          rows: [{ label: "x", value: { "@path": "/x" } }],
        },
      ],
      { x: 1 },
    );
    act(() => {});
    expect(v.reportError).not.toHaveBeenCalled();
  });
});
