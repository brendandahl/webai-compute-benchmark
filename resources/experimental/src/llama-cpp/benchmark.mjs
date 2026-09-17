// Copyright 2026 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

/**
 * Experimental llama.cpp benchmark using @wllama/wllama and the Gemma model.
 *
 * wllama ships a single `wllama.wasm` containing both the CPU (WebAssembly
 * SIMD) and the WebGPU ggml backends, and picks between them at load time
 * based on `n_gpu_layers`. Each suite pins that value and asserts up front
 * that its backend is actually available, so a browser that cannot run the
 * requested backend fails instead of quietly measuring the other one.
 */

import { Wllama } from "@wllama/wllama/esm/index.js";
import { BenchmarkConnector } from "speedometer-utils/benchmark.mjs";
import { createSubIteratedSuite } from "speedometer-utils/helpers.mjs";
import { params } from "speedometer-utils/params.mjs";

const weightsPath = "../models/llama-cpp/gemma-3-270m-it-Q8_0.gguf";
const wasmPath = "resources/wasm/wllama.wasm";

const ONE_MB = 1024 * 1024;
const TEN_MB = 10 * ONE_MB;

const BACKENDS = {
  wasm: {
    id: "wasm",
    label: "WebAssembly SIMD (CPU)",
    // 0 is special-cased by wllama: besides telling llama.cpp not to offload,
    // it stubs `navigator.gpu.requestAdapter()` inside the worker to return
    // null, so the WebGPU backend cannot take over.
    nGpuLayers: 0,
  },
  webgpu: {
    id: "webgpu",
    label: "WebGPU",
    // Offload every layer. This matches wllama's own default.
    nGpuLayers: 99999,
    async assertAvailable() {
      // Checking `navigator.gpu` alone is not enough: Chrome exposes the
      // object even when no adapter can be created (e.g. headless without a
      // GPU), and in that case ggml silently falls back to the CPU backend
      // and the suite would report CPU timings as if they were WebGPU.
      const adapter = navigator.gpu
        ? await navigator.gpu.requestAdapter()
        : null;
      if (!adapter) {
        throw new Error(
          "WebGPU is not available in this browser " +
            (navigator.gpu
              ? "(navigator.gpu exposed no adapter)"
              : "(navigator.gpu is undefined)") +
            ", so this benchmark cannot measure the WebGPU backend.",
        );
      }
    },
  },
};

let lastLogged = -1;
function logDownloadProgress({ loaded, total }) {
  if (total) {
    const percent = Math.floor((loaded / total) * 100);
    if (percent !== lastLogged) {
      console.log(`Downloading model: ${percent}%`);
      lastLogged = percent;
    }
  } else {
    const currentMb = Math.floor(loaded / TEN_MB);
    if (currentMb !== lastLogged) {
      console.log(`Downloading model: ${Math.floor(loaded / ONE_MB)} MB`);
      lastLogged = currentMb;
    }
  }
}

// `load_tensors:       WebGPU model buffer size =  271.81 MiB`, where the name
// is the ggml buffer type holding the weights (`CPU`, `WebGPU`, ...).
const WEIGHTS_BUFFER_LOG = /load_tensors:\s+(\S+)\s+model buffer size/;

/**
 * Confirms the weights landed in the expected buffer type.
 *
 * An available adapter does not guarantee ggml used it: if the WebGPU backend
 * fails to initialize it falls back to CPU without raising an error, and the
 * suite would report CPU timings as WebGPU ones.
 */
function assertWeightsOnExpectedBackend(backend, nativeLogs) {
  const buffers = nativeLogs
    .map((line) => line.match(WEIGHTS_BUFFER_LOG)?.[1])
    .filter(Boolean);
  if (buffers.length === 0) {
    throw new Error(
      "Could not find llama.cpp `load_tensors` output, so the backend in use " +
        "could not be confirmed (did the log format change?).",
    );
  }
  const onGpu = buffers.some((name) => /webgpu/i.test(name));
  if (onGpu !== (backend.id === BACKENDS.webgpu.id)) {
    throw new Error(
      `Expected llama.cpp to run on the ${backend.label} backend, but the ` +
        `model weights are in: ${buffers.join(", ")}.`,
    );
  }
  return buffers;
}

class LlamaCppBenchmark {
  constructor(backend) {
    this.backend = backend;
    this.wllama = null;
  }

  async init() {
    console.log(
      `Initializing llama.cpp runtime (${this.backend.label} backend)...`,
    );
    // Check before downloading ~290 MB of weights.
    await this.backend.assertAvailable?.();

    const wasmUrl = new URL(wasmPath, window.location.href).href;
    // Keep llama.cpp's native log lines so the backend can be confirmed after
    // loading; they are still printed, as they are without a custom logger.
    const nativeLogs = [];
    const tee =
      (method) =>
      (...args) => {
        nativeLogs.push(args.map(String).join(" "));
        console[method](...args);
      };
    this.wllama = new Wllama(
      { default: wasmUrl },
      {
        logger: {
          debug: tee("debug"),
          log: tee("log"),
          warn: tee("warn"),
          error: tee("error"),
        },
      },
    );

    const modelUrl = new URL(weightsPath, window.location.href).href;
    const maxThreads = Math.min(navigator.hardwareConcurrency || 4, 16);
    console.log("Downloading model and initializing llama.cpp context...");
    await this.wllama.loadModelFromUrl(modelUrl, {
      progressCallback: logDownloadProgress,
      n_cache_reuse: 0,
      n_threads: maxThreads,
      n_gpu_layers: this.backend.nGpuLayers,
    });
    const buffers = assertWeightsOnExpectedBackend(this.backend, nativeLogs);
    console.log(
      `llama.cpp initialized on the ${this.backend.label} backend ` +
        `(multithread: ${this.wllama.isMultithread()}, ` +
        `threads: ${this.wllama.getNumThreads()}, ` +
        `weight buffers: ${buffers.join(", ")}).`,
    );
    if (this.backend.id === BACKENDS.wasm.id && !this.wllama.isMultithread()) {
      console.warn(
        "llama.cpp is running single-threaded. Multi-threaded wasm requires " +
          "cross-origin isolation (COOP/COEP headers); timings will not be " +
          "comparable to a cross-origin isolated run.",
      );
    }
  }

  async run() {
    const sentence = "Max 100 word response. Why is the sky blue?";
    console.log("Generating...");
    console.time("llama-cpp-generation");
    const result = await this.wllama.createChatCompletion({
      messages: [{ role: "user", content: sentence }],
      cache_prompt: false,
    });
    console.timeEnd("llama-cpp-generation");
    console.log(result?.choices?.[0]?.message?.content ?? result);
  }
}

const appVersion = "0.1.0";

/**
 * Wires up the benchmark for one of the backends in `BACKENDS`.
 *
 * @param {keyof typeof BACKENDS} backendId
 */
export async function initializeLlamaCppBenchmark(backendId) {
  const backend = BACKENDS[backendId];
  if (!backend) {
    throw new Error(
      `Unknown llama.cpp backend "${backendId}", expected one of: ${Object.keys(BACKENDS).join(", ")}`,
    );
  }

  const appName = `Llama.cpp-${backend.id}`;
  try {
    const benchmark = new LlamaCppBenchmark(backend);
    await benchmark.init();

    /*--------- Running test suites ---------*/
    const suites = {
      default: createSubIteratedSuite(benchmark, params.subIterationCount),
    };

    const benchmarkConnector = new BenchmarkConnector(
      suites,
      appName,
      appVersion,
    );
    benchmarkConnector.connect();
  } catch (error) {
    console.error(`Failed to initialize ${appName} benchmark:`, error);
    throw error;
  }
}
