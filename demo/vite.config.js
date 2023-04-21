import { defineConfig } from "vite";
import path from "path";
import react from "@vitejs/plugin-react";

export default defineConfig({
    server: { open: true },
    plugins: [react()],
    assetsInclude: [
        "**/*.onnx",
        "**/vocab.json",
        "**/*.wasm",
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
});
