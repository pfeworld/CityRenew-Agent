import axios from "axios";

// baseURL 可配置：
// - 若设置 VITE_API_BASE_URL，则直连该地址。
// - 否则使用 "/"，开发期由 Vite proxy 将 /api 代理到本地后端。
// 红线：仅访问本地后端，绝不调用任何外部 API / 大模型。
export const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || "/";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export default client;
