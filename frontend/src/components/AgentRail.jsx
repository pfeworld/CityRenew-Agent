import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Dropdown, Modal, Input, message as antdMessage } from "antd";
import {
  Bot,
  Plus,
  Search,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Archive,
  ArchiveRestore,
  Share2,
  Trash2,
} from "lucide-react";
import {
  listConversations,
  renameConversation,
  deleteConversation,
  pinConversation,
  archiveConversation,
  shareConversation,
} from "../api/agentApi.js";

export default function AgentRail() {
  const navigate = useNavigate();
  const location = useLocation();
  const [convs, setConvs] = useState([]);
  const [query, setQuery] = useState("");
  const [includeArchived, setIncludeArchived] = useState(false);
  const [rename, setRename] = useState({ open: false, id: null, title: "" });
  const [share, setShare] = useState({ open: false, link: "", summary: "" });
  const queryRef = useRef("");

  const refresh = useCallback(async () => {
    const res = await listConversations({ query: queryRef.current, includeArchived });
    if (res.ok) setConvs(res.data.conversations || []);
  }, [includeArchived]);

  useEffect(() => {
    refresh();
    const sync = () => refresh();
    window.addEventListener("cr-conv-updated", sync);
    return () => window.removeEventListener("cr-conv-updated", sync);
  }, [refresh]);

  const activeConv = (() => {
    try {
      return new URLSearchParams(location.search).get("c");
    } catch {
      return null;
    }
  })();

  const newChat = () => navigate("/agent?new=" + Date.now());

  const onSearch = (v) => {
    setQuery(v);
    queryRef.current = v;
    refresh();
  };

  const doRename = async () => {
    const title = rename.title.trim();
    if (!title) return setRename({ open: false, id: null, title: "" });
    const res = await renameConversation(rename.id, title);
    setRename({ open: false, id: null, title: "" });
    if (res.ok) {
      antdMessage.success("已重命名");
      refresh();
    } else antdMessage.error(res.message || "重命名失败");
  };

  const confirmDelete = (c) => {
    Modal.confirm({
      title: "删除会话",
      content: `确定删除「${c.title || "该会话"}」吗？该操作不可恢复。`,
      okText: "删除",
      okType: "danger",
      cancelText: "取消",
      onOk: async () => {
        const res = await deleteConversation(c.id);
        if (res.ok) {
          antdMessage.success("已删除");
          if (activeConv === c.id) navigate("/agent?new=" + Date.now());
          refresh();
        } else antdMessage.error(res.message || "删除失败");
      },
    });
  };

  const togglePin = async (c) => {
    const res = await pinConversation(c.id, !c.pinned);
    if (res.ok) refresh();
    else antdMessage.error(res.message || "操作失败");
  };

  const toggleArchive = async (c) => {
    const res = await archiveConversation(c.id, !c.archived);
    if (res.ok) {
      antdMessage.success(c.archived ? "已取消归档" : "已归档");
      refresh();
    } else antdMessage.error(res.message || "操作失败");
  };

  const doShare = async (c) => {
    const res = await shareConversation(c.id);
    if (!res.ok) return antdMessage.error(res.message || "暂时无法分享");
    const link = `${window.location.origin}${res.data.link}`;
    setShare({ open: true, link, summary: res.data.summary || "" });
  };

  const copy = async (text, tip) => {
    try {
      await navigator.clipboard.writeText(text);
      antdMessage.success(tip);
    } catch {
      antdMessage.warning("当前浏览器不支持自动复制，请手动选择复制。");
    }
  };

  const menuItems = (c) => [
    { key: "rename", label: "重命名", icon: <Pencil size={14} /> },
    {
      key: "pin",
      label: c.pinned ? "取消置顶" : "置顶",
      icon: c.pinned ? <PinOff size={14} /> : <Pin size={14} />,
    },
    {
      key: "archive",
      label: c.archived ? "取消归档" : "归档",
      icon: c.archived ? <ArchiveRestore size={14} /> : <Archive size={14} />,
    },
    { key: "share", label: "分享", icon: <Share2 size={14} /> },
    { type: "divider" },
    { key: "delete", label: "删除", icon: <Trash2 size={14} />, danger: true },
  ];

  const onMenu = (c, key) => {
    if (key === "rename") setRename({ open: true, id: c.id, title: c.title || "" });
    else if (key === "pin") togglePin(c);
    else if (key === "archive") toggleArchive(c);
    else if (key === "share") doShare(c);
    else if (key === "delete") confirmDelete(c);
  };

  return (
    <aside className="ag-rail">
      <div className="ag-rail__identity">
        <div className="ag-rail__avatar">
          <Bot size={22} strokeWidth={2.1} />
        </div>
        <div style={{ minWidth: 0 }}>
          <div className="ag-rail__name">CityRenew Agent</div>
          <div className="ag-rail__role">城市更新策划顾问</div>
        </div>
      </div>

      <button className="ag-rail__new" onClick={newChat}>
        <Plus size={16} /> 新建对话
      </button>

      <div className="ag-rail__search">
        <Search size={14} />
        <input
          value={query}
          placeholder="搜索历史会话"
          onChange={(e) => onSearch(e.target.value)}
        />
      </div>

      <div className="ag-rail__history">
        <div className="ag-rail__label">
          <span>历史会话</span>
          <button className="ag-rail__archtoggle" onClick={() => setIncludeArchived((v) => !v)}>
            {includeArchived ? "隐藏归档" : "显示归档"}
          </button>
        </div>
        <div className="ag-rail__list">
          {convs.length === 0 ? (
            <div style={{ fontSize: 12, color: "#9aa6b8", padding: "6px 10px" }}>
              {query ? "未找到匹配的会话" : "暂无历史会话"}
            </div>
          ) : (
            convs.map((c) => (
              <div
                key={c.id}
                className={`ag-rail__conv ${activeConv === c.id ? "ag-rail__conv--active" : ""} ${
                  c.archived ? "ag-rail__conv--archived" : ""
                }`}
                onClick={() => navigate(`/agent?c=${c.id}`)}
              >
                {c.pinned && <Pin size={12} className="ag-rail__pin" />}
                <div className="ag-rail__conv-title">{c.title || "新的对话"}</div>
                <Dropdown
                  trigger={["click"]}
                  menu={{ items: menuItems(c), onClick: ({ key, domEvent }) => { domEvent.stopPropagation(); onMenu(c, key); } }}
                >
                  <button
                    className="ag-rail__more"
                    onClick={(e) => e.stopPropagation()}
                    aria-label="更多操作"
                  >
                    <MoreHorizontal size={16} />
                  </button>
                </Dropdown>
              </div>
            ))
          )}
        </div>
      </div>

      <Modal
        open={rename.open}
        title="重命名会话"
        okText="保存"
        cancelText="取消"
        onOk={doRename}
        onCancel={() => setRename({ open: false, id: null, title: "" })}
      >
        <Input
          value={rename.title}
          maxLength={40}
          placeholder="输入新的会话名称"
          onChange={(e) => setRename((s) => ({ ...s, title: e.target.value }))}
          onPressEnter={doRename}
        />
      </Modal>

      <Modal
        open={share.open}
        title="分享会话"
        footer={null}
        onCancel={() => setShare({ open: false, link: "", summary: "" })}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>会话链接</div>
            <Input.Search
              value={share.link}
              readOnly
              enterButton="复制链接"
              onSearch={() => copy(share.link, "链接已复制")}
            />
          </div>
          <div>
            <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>会话摘要</div>
            <Input.TextArea value={share.summary} readOnly rows={6} />
            <button className="ag-btn" style={{ marginTop: 8 }} onClick={() => copy(share.summary, "摘要已复制")}>
              复制摘要
            </button>
          </div>
        </div>
      </Modal>
    </aside>
  );
}
