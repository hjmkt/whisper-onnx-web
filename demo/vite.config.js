import { defineConfig } from "vite";
import path from "path";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command, mode }) => {
    if (mode == "production") {
        return {
            server: {
                open: true,
                proxy: {
                    "/ort-wasm-simd.wasm": {
                        target: "/whisper-onnx-web/ort-wasm-simd.wasm",
                        changeOrigin: true,
                    },
                    "/ort-wasm-simd-threaded.wasm": {
                        target: "/whisper-onnx-web/ort-wasm-simd-threaded.wasm",
                        changeOrigin: true,
                    },
                    "/ort-wasm-threaded.wasm": {
                        target: "/whisper-onnx-web/ort-wasm-threaded.wasm",
                        changeOrigin: true,
                    },
                },
            },
            plugins: [react()],
            assetsInclude: [
                "**/*.onnx",
                "**/vocab.json",
                //"**/*.wasm",
                "**/*.gz",
                "**/*.compressed",
            ],
            resolve: {
                alias: {
                    "@core": path.resolve(__dirname, "../src"),
                    "@models": path.resolve(__dirname, "../models"),
                },
                modules: [
                    path.resolve(__dirname, "../src/node_modules"),
                    "node_nodules",
                ],
            },
            build: { sourcemap: true },
        };
    } else {
        return {
            server: {
                open: true,
            },
            plugins: [react()],
            assetsInclude: [
                "**/*.onnx",
                "**/vocab.json",
                //"**/*.wasm",
                "**/*.gz",
                "**/*.compressed",
            ],
            resolve: {
                alias: {
                    "@core": path.resolve(__dirname, "../src"),
                    "@models": path.resolve(__dirname, "../models"),
                },
                modules: [
                    path.resolve(__dirname, "../src/node_modules"),
                    "node_nodules",
                ],
            },
            build: { sourcemap: true },
        };
    }
});
