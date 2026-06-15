import { Outlet } from "react-router-dom";
import { Sparkles } from "lucide-react";
import AgentRail from "./AgentRail.jsx";

// 正式商业智能体外壳（浅色）。顶栏只显示产品身份，
// 不展示任何内部入口 / 模型 / 接入状态 / 训练材料。
export default function AppShell() {
  return (
    <div className="ag-shell">
      <AgentRail />
      <div className="ag-main">
        <header className="ag-topbar">
          <div className="ag-topbar__brand">
            <span className="ag-topbar__logo">
              <Sparkles size={17} strokeWidth={2.2} />
            </span>
            <div>
              <div className="ag-topbar__title">CityRenew Agent</div>
              <div className="ag-topbar__sub">城市更新前期策划智能体</div>
            </div>
          </div>
        </header>
        <main className="ag-content ag-content--full">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
