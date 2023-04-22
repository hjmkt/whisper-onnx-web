import { defineConfig } from "vite";
import path from "path";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command, mode }) => {
    if (mode == "production" || true) {
        return {
            server: {
                fs: {
                    allow: [
                        path.resolve(__dirname, "../src"),
                        path.resolve(__dirname, "src"),
                    ],
                },
            },
            base: "/whisper-onnx-web",
            plugins: [react()],
            assetsInclude: [
                "**/*.onnx",
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
            plugins: [react()],
            assetsInclude: [
                "**/*.onnx",
                "**/vocab.json",
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
