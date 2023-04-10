import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
    server: { open: true },
    plugins: [react()],
    assetsInclude: ["**/*.onnx", "**/vocab.json", "**/*.wasm"],
    build: { sourcemap: true }
});
