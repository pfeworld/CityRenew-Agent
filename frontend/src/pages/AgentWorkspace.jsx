import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { message as antdMessage, Modal, Input } from "antd";
import {
  Sparkles,
  Send,
  Plus,
  Eye,
  Copy,
  FileDown,
  RefreshCw,
  FileText,
  Image as ImageIcon,
  Table as TableIcon,
  ClipboardType,
  X,
} from "lucide-react";
import {
  createConversation,
  getConversation,
  postConversationChat,
  uploadConversationAttachment,
  getReportContent,
  openReportDownload,
} from "../api/agentApi.js";

let _seq = 0;
const nextId = () => `m${Date.now()}_${_seq++}`;

const FIELD_LABELS = {
  name: "项目名称",
  address: "项目地址",
  district: "所在区",
  land_use: "用地性质",
  build_year: "建成年代",
  project_area: "用地面积",
  building_area: "建筑面积",
  update_demand: "现状问题",
  expected_direction: "更新目标",
};
const labelFields = (fields) =>
  (fields || []).map((f) => FIELD_LABELS[f] || f).join("、");

const PLUS_MENU = [
  { key: "file", label: "上传项目文件", icon: FileText, accept: ".doc,.docx,.pdf,.txt,.md,.ppt,.pptx" },
  { key: "image", label: "上传图片", icon: ImageIcon, accept: "image/*" },
  { key: "table", label: "上传表格", icon: TableIcon, accept: ".xls,.xlsx,.csv" },
  { key: "paste", label: "粘贴文本资料", icon: ClipboardType, accept: "" },
];

function notifyConvUpdated() {
  window.dispatchEvent(new Event("cr-conv-updated"));
}

// 分析中阶段提示（流程描述，非数据；轮播以呈现“正在分析”动效）
const ANALYZING_STAGES = [
  "正在解析项目位置与三圈层范围…",
  "归集 POI 配套与区位指标…",
  "分析人口客群与房价空间现状…",
  "研判产业经济与项目类型…",
  "组织分析结论与更新建议…",
];

function AnalyzingBubble() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(
      () => setStep((v) => (v + 1) % ANALYZING_STAGES.length),
      1600
    );
    return () => clearInterval(t);
  }, []);
  return (
    <div className="ag-analyzing">
      <span className="ag-analyzing__orb">
        <Sparkles size={15} />
      </span>
      <div className="ag-analyzing__main">
        <div className="ag-analyzing__title">
          正在分析中
          <span className="ag-analyzing__dots">
            <i />
            <i />
            <i />
          </span>
        </div>
        <div className="ag-analyzing__stage" key={step}>
          {ANALYZING_STAGES[step]}
        </div>
        <div className="ag-analyzing__bar">
          <span />
        </div>
      </div>
    </div>
  );
}

export default function AgentWorkspace() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [focus, setFocus] = useState(false);
  const [stage, setStage] = useState("待输入资料");
  const [reportReady, setReportReady] = useState(false);
  const [reportId, setReportId] = useState(null);
  const [attachments, setAttachments] = useState([]); // [{id,file,name}]
  const [plusOpen, setPlusOpen] = useState(false);
  const [paste, setPaste] = useState({ open: false, text: "" });
  const [preview, setPreview] = useState({ open: false, loading: false, title: "", text: "", blocks: [] });
  const bottomRef = useRef(null);
  const fileRef = useRef(null);
  const initRef = useRef({ c: null, nw: null });
  const submittingRef = useRef(false); // 同步锁：防止快速双触发重复创建会话
  const previewScrollRef = useRef(null); // 预览滚动容器：每次打开/切换报告回到顶部

  const hasConversation = messages.length > 0;

  useEffect(() => {
    let alive = true;
    const c = searchParams.get("c");
    const nw = searchParams.get("new");

    // 新建/空白态：仅进入前端草稿，不创建后端持久化会话（根因修复）。
    function resetToDraft() {
      if (!alive) return;
      initRef.current = { c: null, nw };
      setConversationId(null);
      setMessages([]);
      setStage("待输入资料");
      setReportReady(false);
      setReportId(null);
      setAttachments([]);
      notifyConvUpdated();
    }

    async function loadExisting(id) {
      const res = await getConversation(id);
      if (alive && res.ok) {
        initRef.current = { c: id, nw };
        setConversationId(id);
        const mapped = (res.data.messages || []).map((m) => ({
          id: nextId(),
          role: m.role === "assistant" ? "bot" : "user",
          text: m.text,
        }));
        if (res.data.report_ready && res.data.report_id) {
          for (let i = mapped.length - 1; i >= 0; i -= 1) {
            if (mapped[i].role === "bot") {
              mapped[i] = { ...mapped[i], report: { ready: true, report_id: res.data.report_id } };
              break;
            }
          }
        }
        setMessages(mapped);
        setStage(res.data.stage || "待输入资料");
        setReportReady(!!res.data.report_ready);
        setReportId(res.data.report_id || null);
        setAttachments([]);
      } else if (alive) {
        resetToDraft();
      }
    }

    if (nw && nw !== initRef.current.nw) {
      resetToDraft();
    } else if (c && c !== initRef.current.c && c !== conversationId) {
      loadExisting(c);
    } else if (!c && !nw && initRef.current.c === null) {
      resetToDraft();
    }

    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  useEffect(() => {
    if (hasConversation) bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, hasConversation]);

  // 预览弹窗：每次打开或切换到新报告内容时，滚动位置回到顶部。
  useEffect(() => {
    if (preview.open && !preview.loading && previewScrollRef.current) {
      previewScrollRef.current.scrollTop = 0;
    }
  }, [preview.open, preview.loading, preview.title, preview.text]);

  const syncMeta = useCallback(
    (data) => {
      setStage(data.stage || "待输入资料");
      setReportReady(!!data.report?.ready);
      setReportId(data.report?.report_id || null);
      if (data.conversation_id && !searchParams.get("c")) {
        initRef.current = { c: data.conversation_id, nw: searchParams.get("new") };
        setSearchParams({ c: data.conversation_id }, { replace: true });
      }
      notifyConvUpdated();
    },
    [searchParams, setSearchParams]
  );

  // 纯对话发送（不含附件处理）；cidOverride 用于草稿首发时刚创建的会话。
  const runChat = useCallback(
    async (text, cidOverride) => {
      const userText = (text ?? "").trim();
      const cid = cidOverride || conversationId;
      if (!userText || !cid) return;
      const botId = nextId();
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", text: userText },
        { id: botId, role: "bot", loading: true },
      ]);
      const res = await postConversationChat({ conversationId: cid, message: userText });
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId
            ? res.ok
              ? {
                  ...m,
                  loading: false,
                  text: res.data.reply,
                  suggestions: res.data.suggestions || [],
                  report: res.data.report,
                }
              : { ...m, loading: false, error: res.message || "请求失败，请稍后重试。" }
            : m
        )
      );
      if (res.ok) syncMeta(res.data);
    },
    [conversationId, syncMeta]
  );

  // 发送：先处理已选附件（上传 → 进入上下文 → 智能体总结），再发送文本
  const submit = useCallback(async () => {
    if (busy || submittingRef.current) return;
    const text = input.trim();
    const atts = attachments;
    if (!text && atts.length === 0) return; // 空消息：不创建会话、不发请求
    submittingRef.current = true;
    setBusy(true);
    setPlusOpen(false);

    // 草稿首发：此时才创建后端会话并取得真实 cid（不再进页面就建库）。
    let cid = conversationId;
    if (!cid) {
      const created = await createConversation();
      if (!created.ok) {
        antdMessage.error(created.message || "暂时无法开始对话，请稍后重试。");
        setBusy(false);
        submittingRef.current = false;
        return;
      }
      cid = created.data.conversation_id;
      setConversationId(cid);
    }

    if (atts.length) {
      const names = atts.map((a) => a.name).join("、");
      setMessages((prev) => [...prev, { id: nextId(), role: "user", text: `已上传：${names}` }]);
      setAttachments([]);
      const sumId = nextId();
      setMessages((prev) => [...prev, { id: sumId, role: "bot", loading: true }]);
      const summaries = [];
      for (const a of atts) {
        // eslint-disable-next-line no-await-in-loop
        const res = await uploadConversationAttachment(cid, a.file);
        if (res.ok) {
          const f = res.data.extracted_fields || [];
          summaries.push(
            f.length
              ? `${a.name}（识别到：${labelFields(f)}）`
              : `${a.name}（${res.data.note || "已记录，未识别到结构化要素"}）`
          );
        } else {
          summaries.push(`${a.name}（解析失败：${res.message || "请稍后重试"}）`);
        }
      }
      const summaryText =
        `已读取你上传的 ${atts.length} 份资料：${summaries.join("；")}。我已把其中可用信息纳入本次分析。` +
        (text ? "" : "\n\n你可以补充现状问题与更新目标，或让我开始初步研判。");
      setMessages((prev) => prev.map((m) => (m.id === sumId ? { ...m, loading: false, text: summaryText } : m)));
      // 仅附件、无文本：会话已转正，更新 URL 并刷新历史列表。
      if (!text) {
        if (!searchParams.get("c")) {
          initRef.current = { c: cid, nw: searchParams.get("new") };
          setSearchParams({ c: cid }, { replace: true });
        }
        notifyConvUpdated();
      }
    }

    setInput("");
    if (text) await runChat(text, cid);
    setBusy(false);
    submittingRef.current = false;
  }, [busy, conversationId, input, attachments, runChat, searchParams, setSearchParams]);

  const onSuggestion = useCallback(
    async (s) => {
      if (busy) return;
      setBusy(true);
      if (s.action === "generate_report") await runChat("生成完整报告");
      else if (s.action === "style_lushang") await runChat("参考标杆城市更新案例风格生成策略");
      else await runChat(s.text || s.label);
      setBusy(false);
    },
    [busy, runChat]
  );

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  // 加号菜单：选择类别后再打开对应选择器（不直接弹系统文件框）
  const onPlusItem = (item) => {
    setPlusOpen(false);
    if (item.key === "paste") {
      setPaste({ open: true, text: "" });
      return;
    }
    if (fileRef.current) {
      fileRef.current.accept = item.accept;
      fileRef.current.multiple = true;
      fileRef.current.click();
    }
  };

  const onFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (!files.length) return;
    setAttachments((prev) => [
      ...prev,
      ...files.map((file) => ({ id: nextId(), file, name: file.name })),
    ]);
  };

  const confirmPaste = () => {
    const t = paste.text.trim();
    if (!t) {
      setPaste({ open: false, text: "" });
      return;
    }
    const file = new File([t], `粘贴文本-${new Date().toLocaleTimeString()}.txt`, { type: "text/plain" });
    setAttachments((prev) => [...prev, { id: nextId(), file, name: file.name }]);
    setPaste({ open: false, text: "" });
  };

  const removeAttachment = (id) => setAttachments((prev) => prev.filter((a) => a.id !== id));

  const openPreview = async (rid) => {
    setPreview({ open: true, loading: true, title: "报告预览", text: "", blocks: [] });
    const res = await getReportContent(rid);
    if (res.ok) {
      setPreview({
        open: true,
        loading: false,
        title: res.data.title || "报告预览",
        text: res.data.plain_text || "",
        blocks: res.data.blocks || [],
      });
    } else {
      setPreview({ open: false, loading: false, title: "", text: "", blocks: [] });
      antdMessage.error(res.message || "暂时无法加载报告预览。");
    }
  };

  const copyBody = async (rid) => {
    const res = await getReportContent(rid);
    if (!res.ok) return antdMessage.error(res.message || "暂时无法复制报告正文。");
    try {
      await navigator.clipboard.writeText(res.data.plain_text || "");
      antdMessage.success("报告正文已复制到剪贴板。");
    } catch {
      antdMessage.warning("当前浏览器不支持自动复制，请在预览中手动选择复制。");
    }
  };

  const InputBox = (
    <div className={`ag-input ${focus ? "ag-input--focus" : ""}`}>
      {attachments.length > 0 && (
        <div className="ag-attach">
          {attachments.map((a) => (
            <span key={a.id} className="ag-chip" title={a.name}>
              <FileText size={12} />
              <span className="ag-chip__name">{a.name}</span>
              <button className="ag-chip__x" onClick={() => removeAttachment(a.id)} aria-label="移除附件">
                <X size={12} />
              </button>
            </span>
          ))}
        </div>
      )}
      <textarea
        rows={hasConversation ? 1 : 3}
        value={input}
        placeholder="输入项目地址、地块名称、现状问题或更新目标，例如：上海市黄浦区某地块，想做城市更新"
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={onKeyDown}
        onFocus={() => setFocus(true)}
        onBlur={() => setFocus(false)}
      />
      <div className="ag-input__bar">
        <div className="ag-plus-wrap">
          <button
            className="ag-input__plus"
            title="添加资料"
            disabled={busy}
            onClick={() => setPlusOpen((v) => !v)}
          >
            <Plus size={16} />
          </button>
          {plusOpen && (
            <>
              <div className="ag-plus-mask" onClick={() => setPlusOpen(false)} />
              <div className="ag-plusmenu">
                {PLUS_MENU.map((item) => (
                  <button key={item.key} className="ag-plusmenu__item" onClick={() => onPlusItem(item)}>
                    <item.icon size={15} /> {item.label}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
        <span className="ag-input__hint">添加项目资料 · Enter 发送 · Shift+Enter 换行</span>
        <button
          className="ag-input__send"
          disabled={busy || (!input.trim() && attachments.length === 0)}
          onClick={submit}
        >
          <Send size={14} /> 发送
        </button>
      </div>
      <input ref={fileRef} type="file" hidden onChange={onFileChange} />
    </div>
  );

  return (
    <div className="ag-agent">
      <div className="ag-chatcol">
        <div className="ag-chatscroll ag-scroll">
          {!hasConversation ? (
            <div className="ag-empty">
              <span className="ag-empty__badge">
                <Sparkles size={13} /> 城市更新前期策划智能体
              </span>
              <h1 className="ag-empty__title">
                你好，我是 <span>CityRenew</span>
              </h1>
              <p className="ag-empty__sub">
                告诉我项目所在城市与地址、现状问题和更新目标，我会完成项目研判、区位与客群分析、房价与产业诊断，
                并为你生成城市更新前期策划报告。资料不全时我会主动追问，你也可以点击输入框左侧的加号添加项目资料。
              </p>
              {InputBox}
            </div>
          ) : (
            <div className="ag-thread">
              {messages.map((m) =>
                m.role === "user" ? (
                  <div key={m.id} className="ag-msg-user">
                    {m.text}
                  </div>
                ) : (
                  <div key={m.id}>
                    {m.error ? (
                      <div className="ag-msg-error">{m.error}</div>
                    ) : (
                      <div className="ag-msg-bot">
                        {m.loading ? (
                          <AnalyzingBubble />
                        ) : (
                          <>
                            <div className="ag-msg-bot__text">{m.text}</div>
                            {m.report?.ready && (
                              <div className="ag-reportcard">
                                <div className="ag-reportcard__head">
                                  <Sparkles size={15} /> 前策报告已生成
                                </div>
                                <div className="ag-reportcard__desc">
                                  已根据当前项目资料生成城市更新前期策划报告。你可以预览内容、复制正文，或下载 Word / 导出 PDF。
                                </div>
                                <div className="ag-reportcard__actions">
                                  <button className="ag-btn ag-btn--primary" onClick={() => openPreview(m.report.report_id)}>
                                    <Eye size={14} /> 预览报告
                                  </button>
                                  <button className="ag-btn" onClick={() => copyBody(m.report.report_id)}>
                                    <Copy size={14} /> 复制正文
                                  </button>
                                  <button className="ag-btn" onClick={() => openReportDownload(m.report.report_id, "docx")}>
                                    <FileDown size={14} /> 下载 Word
                                  </button>
                                  <button className="ag-btn" onClick={() => openReportDownload(m.report.report_id, "pdf")}>
                                    <FileDown size={14} /> 导出 PDF
                                  </button>
                                  <button className="ag-btn" onClick={() => onSuggestion({ text: "请继续优化这份报告" })}>
                                    <RefreshCw size={14} /> 继续优化
                                  </button>
                                </div>
                              </div>
                            )}
                            {(() => {
                              const chips = (m.suggestions || []).filter(
                                (s) => s.action !== "export_docx" && s.action !== "export_pdf"
                              );
                              return chips.length > 0 ? (
                                <div className="ag-suggest">
                                  {chips.map((s, i) => (
                                    <button key={i} className="ag-suggest__chip" onClick={() => onSuggestion(s)}>
                                      {s.label}
                                    </button>
                                  ))}
                                </div>
                              ) : null;
                            })()}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
        {hasConversation && <div className="ag-dock">{InputBox}</div>}
      </div>

      <Modal
        open={paste.open}
        title="粘贴文本资料"
        okText="加入附件"
        cancelText="取消"
        onOk={confirmPaste}
        onCancel={() => setPaste({ open: false, text: "" })}
      >
        <Input.TextArea
          rows={8}
          value={paste.text}
          placeholder="把项目相关的文字资料粘贴到这里，发送后将作为附件纳入分析。"
          onChange={(e) => setPaste((p) => ({ ...p, text: e.target.value }))}
        />
      </Modal>

      <Modal
        open={preview.open}
        title={preview.title}
        onCancel={() => setPreview((p) => ({ ...p, open: false }))}
        footer={null}
        width={760}
        destroyOnClose
        styles={{ body: { padding: 0 } }}
      >
        <div ref={previewScrollRef} className="ag-preview-scroll">
        {preview.loading ? (
          <div style={{ padding: 24, color: "#6b7280" }}>正在加载报告内容…</div>
        ) : preview.blocks && preview.blocks.length ? (
          <div className="ag-doc">
            {preview.blocks.map((b, i) => {
              if (b.type === "table") {
                const [head, ...body] = b.rows || [];
                return (
                  <table className="ag-doc__table" key={i}>
                    {head && (
                      <thead>
                        <tr>{head.map((c, j) => <th key={j}>{c}</th>)}</tr>
                      </thead>
                    )}
                    <tbody>
                      {body.map((r, ri) => (
                        <tr key={ri}>{r.map((c, ci) => <td key={ci}>{c}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                );
              }
              const lvl = b.level || 0;
              if (lvl === 1) return <h1 className="ag-doc__h1" key={i}>{b.text}</h1>;
              if (lvl === 2) return <h2 className="ag-doc__h2" key={i}>{b.text}</h2>;
              if (lvl === 3) return <h3 className="ag-doc__h3" key={i}>{b.text}</h3>;
              return <p className="ag-doc__p" key={i}>{b.text}</p>;
            })}
          </div>
        ) : (
          <pre className="ag-preview__body">{preview.text}</pre>
        )}
        </div>
      </Modal>
    </div>
  );
}
