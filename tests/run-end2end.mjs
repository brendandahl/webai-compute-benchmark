#! /usr/bin/env node

import assert from "assert";
import testSetup from "./helper.mjs";
import { benchmarkConfigurator } from "../resources/benchmark-configurator.mjs";
import { defaultParams } from "../resources/shared/params.mjs";

const HELP = `
This script runs end2end tests by invoking the benchmark via the main page in /index.html.
`.trim();

const ONE_MINUTE_IN_MS = 60000;

const { driver, PORT, stop } = await testSetup(HELP);

// Running all of the benchmarks is very slow (especially when the GPU is emulated). To run the
// tests faster we run all of the Wasm benchmarks, and only a few GPU tests to cover most of
// the common code. To run all benchmarks, enable this.
const RUN_FULL_SUITE = false;
let tags = "wasm,gpu-test-suite";
let suites = benchmarkConfigurator.suites.filter((suite) => !suite.url.includes("/experimental/") && suite.tags.some((tag) => tag === "wasm" || tag === "gpu-test-suite"));
let timeout = 10 * ONE_MINUTE_IN_MS;

if (RUN_FULL_SUITE) {
    tags = "all";
    suites = benchmarkConfigurator.suites;
    timeout = 20 * ONE_MINUTE_IN_MS;
}

async function testPage(url) {
    console.log(`\nTesting: ${url}`);
    await driver.get(`http://localhost:${PORT}/${url}`);

    await driver.executeAsyncScript((callback) => {
        if (globalThis.benchmarkClient) {
            callback();
        } else {
            globalThis.addEventListener("BenchmarkReady", () => callback(), { once: true });
        }
    });

    await driver.executeScript((defaultSubIterationCount) => {
        globalThis._e2eQueue = [];
        globalThis._e2eWaiter = null;
        const push = (event) => {
            if (globalThis._e2eWaiter) {
                globalThis._e2eWaiter([event]);
                globalThis._e2eWaiter = null;
            } else {
                globalThis._e2eQueue.push(event);
            }
        };

        const client = globalThis.benchmarkClient;
        const defaultSubIters = Number(new URLSearchParams(location.search).get("subIterationCount") ?? defaultSubIterationCount);
        const getSubIters = (suite) => suite.steps?.length ?? defaultSubIters;
        let step = 0;
        let prepStr = "";
        const pushLive = (suite, status = `${prepStr}running...`) => push({ live: `    ⏳ ${suite.name} [sub-iter ${step + 1}/${getSubIters(suite)}] (${status})` });

        client.willStartIteration = (i, count) => push({ log: `  Iteration ${i + 1}/${count}` });

        const origStartSuite = client.willStartSuite.bind(client);
        client.willStartSuite = (suite) => {
            origStartSuite(suite);
            step = 0;
            prepStr = "";
            pushLive(suite, "preparing...");
        };
        client.didFinishSuitePrepare = (suite, prepare) => {
            prepStr = `prepare: ${prepare.toFixed(1)}ms, `;
            pushLive(suite);
        };
        client.didFinishStep = (suite) => {
            if (++step < getSubIters(suite)) {
                pushLive(suite);
            }
        };

        const origFinishSuite = client.didFinishSuite.bind(client);
        client.didFinishSuite = (suite, results) => {
            origFinishSuite(suite);
            const stepTimes = Object.values(results.steps).map((s) => `${s.total.toFixed(1)}ms`);
            const breakdown = stepTimes.length > 1 ? ` [${stepTimes.join(", ")}]` : "";
            const subLabel = stepTimes.length > 1 ? ` (${stepTimes.length} sub-iters)` : "";
            push({ log: `    \x1b[32m✓\x1b[0m ${suite.name}${subLabel} (${prepStr}run: ${results.total.toFixed(1)}ms${breakdown})` });
        };

        const origFailSuite = client.didFailSuite.bind(client);
        client.didFailSuite = (suite, err) => {
            origFailSuite(suite);
            const msg = err?.stack || err?.message || String(err);
            push({ error: `Suite ${suite.name} failed: ${msg}` });
        };

        globalThis.addEventListener("BenchmarkDone", () => push({ done: client.metrics }), { once: true });
        globalThis.addEventListener("error", (e) => push({ error: e.message + (e.error?.stack ?? "") }));
        globalThis.addEventListener("unhandledrejection", (e) => push({ error: (e.reason?.toString?.() ?? String(e.reason)) + (e.reason?.stack ?? "") }));
        client.start();
    }, defaultParams.subIterationCount);

    const isTTY = Boolean(process.stdout.isTTY);
    let lastLive = "";
    let metrics = null;
    const println = (text) => process.stdout.write(isTTY ? `\r\x1b[2K${text}\n` : `${text}\n`);

    try {
        while (!metrics) {
            const events = await driver.executeAsyncScript((cb) => {
                if (globalThis._e2eQueue.length) {
                    cb(globalThis._e2eQueue.splice(0));
                } else {
                    globalThis._e2eWaiter = cb;
                }
            });
            for (const { live, log, error, done } of events) {
                if (live) {
                    lastLive = live;
                    if (isTTY) {
                        process.stdout.write(`\r\x1b[2K${live}`);
                    }
                }
                if (log) {
                    lastLive = "";
                    println(log);
                }
                if (error) {
                    throw new Error(error);
                }
                if (done) {
                    metrics = done;
                }
            }
        }
    } catch (err) {
        if (lastLive) {
            println(`${lastLive.replace("⏳", "\x1b[31m✖\x1b[0m")} - failed`);
        }
        throw err;
    }

    validateMetrics(metrics);
    return metrics;
}

function validateMetrics(metrics) {
    for (const [name, metric] of Object.entries(metrics)) {
        validateMetric(name, metric);
    }
    assert(metrics["Wasm-Geomean"]?.mean > 0 || metrics["WebGPU-Geomean"]?.mean > 0);
    assert(metrics["Wasm-Score"]?.mean > 0 || metrics["WebGPU-Score"]?.mean > 0);
}

function validateMetric(name, metric) {
    assert(metric.name === name);
    assert(metric.mean >= 0);
}

async function testIterations() {
    const iterationCount = 2;
    const subIterationCount = 1;
    const metrics = await testPage(`index.html?iterationCount=${iterationCount}&subIterationCount=${subIterationCount}&tags=${tags}`);
    suites.forEach((suite) => {
        if (suite.enabled) {
            const metric = metrics[suite.name];
            assert(metric, `Missing suite result for ${suite.name}`);
            assert(metric.values.length === iterationCount);
        } else {
            assert(!(suite.name in metrics));
        }
    });
    if (metrics["Wasm-Geomean"]?.mean > 0) {
        assert(metrics["Wasm-Geomean"].values.length === iterationCount);
        assert(metrics["Wasm-Score"].values.length === iterationCount);
    }
    if (metrics["WebGPU-Geomean"]?.mean > 0) {
        assert(metrics["WebGPU-Geomean"].values.length === iterationCount);
        assert(metrics["WebGPU-Score"].values.length === iterationCount);
    }
}

async function testSubIterations() {
    const testSuites = ["Image-Classification-LiteRT.js-wasm", "Feature-Extraction-wasm"];

    let suites = benchmarkConfigurator.suites.filter((suite) => testSuites.includes(suite.name));
    const iterationCount = 1;
    const subIterationCount = 3;
    // URL with suites specified
    const params = [`iterationCount=${iterationCount}`, `subIterationCount=${subIterationCount}`, `suites=${testSuites.join(",")}`];
    const metrics = await testPage(`index.html?${params.join("&")}`);

    suites.forEach((suite) => {
        const metric = metrics[suite.name];
        assert(metric, `Missing suite result for ${suite.name}`);
        assert(metric.values.length === iterationCount);

        // Verify submetrics generated from steps
        for (let i = 0; i < subIterationCount; i++) {
            // we use some() to find the submetric since the separator might be '/'
            const submetricKey = Object.keys(metrics).find((k) => k.startsWith(suite.name) && k.includes(`sub-iter-${i + 1}`));
            assert(submetricKey, `Missing submetric result ending in sub-iter-${i + 1} for ${suite.name}`);
            const submetric = metrics[submetricKey];
            assert(submetric.values.length === iterationCount);
        }
    });
}

async function testAll() {
    const metrics = await testPage(`index.html?iterationCount=1&subIterationCount=1&tags=${tags}`);
    suites.forEach((suite) => {
        assert(suite.name in metrics);
        const metric = metrics[suite.name];
        assert(metric.values.length === 1);
    });
    if (metrics["Wasm-Geomean"]?.mean > 0) {
        assert(metrics["Wasm-Geomean"].values.length === 1);
        assert(metrics["Wasm-Score"].values.length === 1);
    }
    if (metrics["WebGPU-Geomean"]?.mean > 0) {
        assert(metrics["WebGPU-Geomean"].values.length === 1);
        assert(metrics["WebGPU-Score"].values.length === 1);
    }
}

async function testDeveloperMode() {
    const params = ["developerMode", "iterationCount=1", "warmupBeforeSync=2", "waitBeforeSync=2", "shuffleSeed=123", "suites=Image-Classification-LiteRT.js-wasm"];
    const metrics = await testPage(`index.html?${params.join("&")}`);
    suites.forEach((suite) => {
        if (suite.name === "Image-Classification-LiteRT.js-wasm") {
            const metric = metrics[suite.name];
            assert(metric.values.length === 1);
        } else {
            assert(!(suite.name in metrics));
        }
    });
}

async function test() {
    try {
        benchmarkConfigurator.suites.forEach((suite) => {
            if (suite.tags.includes("default") && suite.tags.includes("experimental")) {
                throw new Error(`Suite "${suite.name}" has both 'default' and 'experimental' tags. Experimental workloads should only have the 'experimental' tag, while stable workloads should have the 'default' tag.`);
            }
        });
        await driver.manage().setTimeouts({ script: timeout });
        await testIterations();
        await testSubIterations();
        await testAll();
        await testDeveloperMode();
        console.log("\nTests complete!");
    } catch (e) {
        console.error("\nTests failed!");
        throw e;
    } finally {
        stop();
    }
}

setImmediate(test);
