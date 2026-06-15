import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// 第12A：后端默认运行于 8023（可通过环境变量覆盖，不写死）。
// - VITE_PROXY_TARGET：开发期 /api、/health 的代理目标（默认 http://localhost:8023）。
// - VITE_API_BASE_URL：前端 axios baseURL（生产/直连场景使用；开发默认走代理）。
// 红线：仅访问本地后端，不调用任何外部 API / 大模型。
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_PROXY_TARGET || "http://localhost:8023";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/health": target,
        "/api": target,
      },
    },
    build: {
      // 第12B：拆分大型第三方依赖，降低首屏单 chunk 体积。
      rollupOptions: {
        output: {
          manualChunks: {
            "vendor-react": ["react", "react-dom", "react-router-dom"],
            "vendor-antd": ["antd", "@ant-design/pro-components", "@ant-design/icons"],
            "vendor-echarts": ["echarts", "echarts-for-react"],
            "vendor-flow": ["reactflow"],
          },
        },
      },
      chunkSizeWarningLimit: 1200,
    },
  };
});
