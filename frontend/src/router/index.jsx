import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Skeleton } from "antd";
import AppShell from "../components/AppShell.jsx";

// 正式前台只暴露智能体工作台，其余历史路径一律重定向到 /agent，
// 不向用户暴露任何内部页面（系统中心 / 模型 / 知识库 / 评估 / 报告后台等）。
const AgentWorkspace = lazy(() => import("../pages/AgentWorkspace.jsx"));

function PageFallback() {
  return (
    <div style={{ padding: 24 }}>
      <Skeleton active paragraph={{ rows: 8 }} />
    </div>
  );
}

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        <Route index element={<Navigate to="/agent" replace />} />
        <Route
          path="agent"
          element={
            <Suspense fallback={<PageFallback />}>
              <AgentWorkspace />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/agent" replace />} />
      </Route>
    </Routes>
  );
}
