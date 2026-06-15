// 第12E：智能体（Agent）只读/交互 API client。
// 红线：
//  1. 前端只调用本地后端 /api/agent/*，绝不直连 DeepSeek，绝不持有 API key。
//  2. health / capabilities 为只读 GET；chat / run-task 为用户主动触发的交互接口（非训练/评测）。
//  3. 接口失败返回标准化错误态，由页面展示状态，不伪造成功。
import client from "./client.js";

async function safeGet(path) {
  try {
    const { data } = await client.get(path);
    return { ok: true, status: "ok", data, message: "" };
  } catch (e) {
    return { ok: false, status: "error", data: null, message: normalizeError(e) };
  }
}

async function safePost(path, body) {
  try {
    const { data } = await client.post(path, body);
    return { ok: true, status: "ok", data, message: "" };
  } catch (e) {
    return { ok: false, status: "error", data: null, message: normalizeError(e) };
  }
}

async function safePatch(path, body) {
  try {
    const { data } = await client.patch(path, body);
    return { ok: true, status: "ok", data, message: "" };
  } catch (e) {
    return { ok: false, status: "error", data: null, message: normalizeError(e) };
  }
}

async function safeDelete(path) {
  try {
    const { data } = await client.delete(path);
    return { ok: true, status: "ok", data, message: "" };
  } catch (e) {
    return { ok: false, status: "error", data: null, message: normalizeError(e) };
  }
}

function normalizeError(e) {
  if (e?.code === "ERR_NETWORK") {
    return "暂时无法连接服务，请稍后重试。";
  }
  if (e?.response?.status) {
    return `${e.response.data?.detail || e.response.statusText || "请求失败，请稍后重试"}`;
  }
  return e?.message || "请求失败，请稍后重试";
}

// 多轮会话（多轮上下文 / 报告状态记忆 / 短输入理解）。
// 项目由后端依据用户输入的地址动态创建，前端不再绑定固定项目。
export const createConversation = () =>
  safePost("/api/agent/conversations", { project_id: null });

export const getConversation = (conversationId) =>
  safeGet(`/api/agent/conversations/${conversationId}`);

// 历史会话管理（后端持久化元数据：列表/搜索/重命名/删除/置顶/归档/分享）。
export const listConversations = ({ query = "", includeArchived = false } = {}) => {
  const qs = new URLSearchParams();
  if (query) qs.set("query", query);
  if (includeArchived) qs.set("include_archived", "true");
  const s = qs.toString();
  return safeGet(`/api/agent/conversations${s ? `?${s}` : ""}`);
};

export const renameConversation = (id, title) =>
  safePatch(`/api/agent/conversations/${id}/rename`, { title });

export const deleteConversation = (id) =>
  safeDelete(`/api/agent/conversations/${id}`);

export const pinConversation = (id, value) =>
  safePost(`/api/agent/conversations/${id}/pin`, { value });

export const archiveConversation = (id, value) =>
  safePost(`/api/agent/conversations/${id}/archive`, { value });

export const shareConversation = (id) =>
  safeGet(`/api/agent/conversations/${id}/share`);

export const postConversationChat = ({ conversationId, message }) =>
  safePost(`/api/agent/conversations/${conversationId}/chat`, { message });

// 加号上传：把附件交给会话所属项目做解析与档案补全（第二阶段已实现）。
export const uploadConversationAttachment = async (conversationId, file) => {
  try {
    const form = new FormData();
    form.append("file", file);
    const { data } = await client.post(
      `/api/agent/conversations/${conversationId}/attachments`,
      form,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
    return { ok: true, data, message: "" };
  } catch (e) {
    return { ok: false, data: null, message: normalizeError(e) };
  }
};

// 读取报告正文（供结果卡片的「预览报告」「复制正文」）。
export const getReportContent = (reportId) =>
  safeGet(`/api/report/${encodeURIComponent(reportId)}/content`);

// 第12G：正式报告生成 / 预览 / 质量 / 下载
export const generateReport = ({ projectId = 1, caseStyle = null } = {}) =>
  safePost("/api/report/generate", { project_id: projectId, case_style: caseStyle });

export const getLatestReport = (projectId = 1) =>
  safeGet(`/api/report/latest?project_id=${projectId}`);

export const reportDocxUrl = (reportId) =>
  `/api/report/${encodeURIComponent(reportId)}/download-docx`;

export const reportPdfUrl = (reportId) =>
  `/api/report/${encodeURIComponent(reportId)}/download-pdf`;

export const openReportDownload = (reportId, fmt) => {
  const url =
    fmt === "pdf" ? reportPdfUrl(reportId) : reportDocxUrl(reportId);
  window.open(url, "_blank");
  return url;
};
