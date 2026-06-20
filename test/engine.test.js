/**
 * QvacEngine decoding-contract tests.
 *
 * These don't need the native @qvac/sdk: we stub `engine.sdk` and pre-seed the
 * model cache, so they assert *how Lantern calls the SDK* (the parameters),
 * which is exactly where the vision-accuracy fix lives.
 *
 * Why this matters: a 500M on-device VLM on default (random) sampling
 * hallucinates "something else entirely". describeImage must pin grounded,
 * greedy decoding (temp 0) so the description stays faithful and reproducible.
 */
import { describe, it, expect } from "vitest";
import { QvacEngine } from "../src/engine/qvac-engine.js";
import { makeConfig, makeLogger } from "./helpers.js";

/** Build an engine with a stubbed SDK that records the last completion params. */
function stubbedEngine() {
  const engine = new QvacEngine(makeConfig(), makeLogger());
  /** @type {any} */
  let lastParams = null;
  engine.sdk = {
    completion(params) {
      lastParams = params;
      return {
        // eslint-disable-next-line require-yield
        tokenStream: (async function* () {
          return;
        })(),
        stats: Promise.resolve({ tokens: 0, tokensPerSecond: 0 }),
      };
    },
  };
  // Pre-seed the model cache so #ensureModel() never calls loadModel().
  engine.models.set("vision", { modelId: "vision-stub", model: "SMOLVLM2", delegated: false });
  engine.models.set("llm", { modelId: "llm-stub", model: "LLAMA", delegated: false });
  return { engine, getParams: () => lastParams };
}

describe("QvacEngine decoding contract", () => {
  it("describeImage pins grounded greedy decoding (temp 0) to stop hallucination", async () => {
    const { engine, getParams } = stubbedEngine();
    await engine.describeImage({ imagePath: "/tmp/x.jpg", prompt: "What is in front of me?" });
    const p = getParams();
    expect(p.generationParams).toBeTruthy();
    expect(p.generationParams.temp).toBe(0);
    expect(p.generationParams.top_k).toBe(1);
  });

  it("describeImage attaches the image and forwards the prompt", async () => {
    const { engine, getParams } = stubbedEngine();
    await engine.describeImage({ imagePath: "/tmp/scene.jpg", prompt: "Describe the scene" });
    const p = getParams();
    expect(p.history[0].content).toBe("Describe the scene");
    expect(p.history[0].attachments[0].path).toContain("scene.jpg");
  });

  it("lets a caller override decoding params when needed", async () => {
    const { engine, getParams } = stubbedEngine();
    await engine.describeImage({ imagePath: "/tmp/x.jpg", prompt: "p", gen: { temp: 0.7, top_k: 40 } });
    expect(getParams().generationParams.temp).toBe(0.7);
  });

  it("chat (LLM phrasing) does not force vision's greedy params", async () => {
    const { engine, getParams } = stubbedEngine();
    await engine.chat({ messages: [{ role: "user", content: "hi" }] });
    // chat passes no generationParams — it keeps the model's default sampling.
    expect(getParams().generationParams).toBeUndefined();
  });
});
