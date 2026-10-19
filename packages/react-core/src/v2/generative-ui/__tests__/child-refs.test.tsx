import type { ReactNode } from "react";
import { fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createGenerativeUI } from "../catalog";
import { createGenerativeUIDispatcher } from "../dispatcher";
import { buildGenerativeUIManifest } from "../manifest";
import { buildCompactGenerativeUISchema } from "../markdown/context";
import { child, childList } from "../refs";
import { standardUI } from "../standard";
import { GenerativeUISurfaceManager } from "../surface-manager";
import { renderSurface } from "../standard/__tests__/render";

function Text({ text }: { text: string | number }) {
  return <span data-testid="text">{String(text)}</span>;
}
function Tabs({ tabs }: { tabs: Array<{ title: string; child: ReactNode }> }) {
  return (
    <div>
      {tabs.map((tab) => (
        <section key={tab.title} data-testid={`tab-${tab.title}`}>
          {tab.child}
        </section>
      ))}
    </div>
  );
}
function Deck({
  slides,
}: {
  slides: Array<{ name: string; items: ReactNode }>;
}) {
  return (
    <div>
      {slides.map((slide) => (
        <section key={slide.name} data-testid={`slide-${slide.name}`}>
          {slide.items}
        </section>
      ))}
    </div>
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
    Tabs: {
      component: Tabs,
      props: z.object({
        tabs: z.array(z.object({ title: z.string(), child: child() })),
      }),
    },
    Deck: {
      component: Deck,
      props: z.object({
        slides: z.array(z.object({ name: z.string(), items: childList() })),
      }),
    },
  },
});

describe("typed child references", () => {
  it("renders a child() inside a list prop, in place of the id", async () => {
    const v = renderSurface(ui, [
      {
        id: "root",
        component: "Tabs",
        tabs: [
          { title: "A", child: "a" },
          { title: "B", child: "b" },
        ],
      },
      { id: "a", component: "Text", text: "panel a" },
      { id: "b", component: "Text", text: "panel b" },
    ]);
    expect((await v.findByTestId("tab-A")).textContent).toBe("panel a");
    expect(v.getByTestId("tab-B").textContent).toBe("panel b");
    expect(v.reportError).not.toHaveBeenCalled();
  });

  it("renders a childList() inside a list prop, with ids and with a template", async () => {
    const v = renderSurface(
      ui,
      [
        {
          id: "root",
          component: "Deck",
          slides: [
            { name: "ids", items: ["x", "y"] },
            { name: "tpl", items: { componentId: "row", path: "/people" } },
          ],
        },
        { id: "x", component: "Text", text: "x" },
        { id: "y", component: "Text", text: "y" },
        { id: "row", component: "Text", text: { "@path": "name" } },
      ],
      { people: [{ name: "Ada" }, { name: "Linus" }] },
    );
    expect((await v.findByTestId("slide-ids")).textContent).toBe("xy");
    expect(v.getByTestId("slide-tpl").textContent).toBe("AdaLinus");
  });

  it("resolves a bound list of tabs from the data model before rendering its children", async () => {
    const v = renderSurface(
      ui,
      [
        { id: "root", component: "Tabs", tabs: { "@path": "/tabs" } },
        { id: "a", component: "Text", text: "from data" },
      ],
      { tabs: [{ title: "A", child: "a" }] },
    );
    expect((await v.findByTestId("tab-A")).textContent).toBe("from data");
  });

  it("shows the prop as componentId in the compact schema and keeps the manifest serializable", () => {
    const manager = new GenerativeUISurfaceManager(ui);
    const dispatcher = createGenerativeUIDispatcher({
      getDeclared: () => ui.functions,
    });
    const manifest = buildGenerativeUIManifest(
      { ui, manager, dispatcher },
      undefined,
    );
    const compact = buildCompactGenerativeUISchema(manifest);
    expect(compact.components.Tabs.props).toEqual({
      tabs: "{ title: string; child: componentId }[]",
    });
    // A description given to child() survives next to the reference.
    const Described = createGenerativeUI({
      id: "d",
      components: {
        Box: {
          component: Column,
          props: z.object({ child: child("The panel") }),
        },
      },
    });
    expect(Described.manifestComponents.Box.props).toEqual({
      type: "object",
      properties: {
        child: { $ref: expect.any(String), description: "The panel" },
      },
      required: ["child"],
      additionalProperties: false,
    });
    expect(compact.components.Deck.props.slides).toBe(
      "{ name: string; items: componentId[] | { componentId: string; path: string } }[]",
    );
    expect(JSON.parse(JSON.stringify(manifest.components.Tabs.props))).toEqual(
      manifest.components.Tabs.props,
    );
  });
});

describe("standard Tabs", () => {
  it("renders Tabs from typed child references", async () => {
    const v = renderSurface(standardUI, [
      {
        id: "root",
        component: "Tabs",
        tabs: [
          { title: "One", child: "a" },
          { title: "Two", child: "b" },
        ],
      },
      { id: "a", component: "Text", text: "panel one" },
      { id: "b", component: "Text", text: "panel two" },
    ]);
    expect((await v.findByRole("tabpanel")).textContent).toBe("panel one");
    fireEvent.click(v.getByRole("tab", { name: "Two" }));
    expect(v.getByRole("tabpanel").textContent).toBe("panel two");
    expect(v.reportError).not.toHaveBeenCalled();
  });
});
