// 第12A：产品级设计系统 tokens（深蓝 / 蓝灰 / 科技青 / 暗金）。
// 统一供 antd ConfigProvider 与自定义组件使用，保证视觉一致、非普通白底后台。

export const palette = {
  // 主背景：深蓝科技政务底
  bgDeep: "#0a1429",
  bgPanel: "#0f1c3a",
  bgPanelAlt: "#13234a",
  bgElevated: "#172a55",
  border: "#23386b",
  borderSoft: "#1c2c54",

  // 文本
  textPrimary: "#eaf1ff",
  textSecondary: "#9fb3d9",
  textMuted: "#6c80a8",

  // 品牌科技青 + 暗金
  primary: "#1f6feb",
  cyan: "#22d3ee",
  gold: "#d4af37",

  // 状态色
  pass: "#3fb950",
  warning: "#d29922",
  fail: "#f85149",
  degraded: "#db6d28",
  info: "#388bfd",
  neutral: "#6c80a8",
};

// antd v5 主题 token（dark 算法在 main.jsx 中启用）
export const antdThemeToken = {
  colorPrimary: palette.primary,
  colorInfo: palette.primary,
  colorSuccess: palette.pass,
  colorWarning: palette.warning,
  colorError: palette.fail,
  colorBgBase: palette.bgDeep,
  colorBgContainer: palette.bgPanel,
  colorBgElevated: palette.bgPanelAlt,
  colorBorder: palette.border,
  colorBorderSecondary: palette.borderSoft,
  colorText: palette.textPrimary,
  colorTextSecondary: palette.textSecondary,
  borderRadius: 10,
  fontSize: 14,
  wireframe: false,
};

// 第12G：正式商业智能体浅色产品风格 token（面向前台用户）。
export const lightPalette = {
  pageBg: "#F6F8FB",
  cardBg: "#FFFFFF",
  sectionBg: "#F2F4F8",
  text: "#172033",
  textSub: "#667085",
  border: "#E6EAF0",
  primary: "#2563EB",
  accent: "#14B8A6",
  softInfo: "#F2F7FF",
};

export const antdLightToken = {
  colorPrimary: lightPalette.primary,
  colorInfo: lightPalette.primary,
  colorSuccess: "#16a34a",
  colorWarning: "#d97706",
  colorError: "#dc2626",
  colorBgBase: "#ffffff",
  colorBgLayout: lightPalette.pageBg,
  colorBgContainer: lightPalette.cardBg,
  colorBgElevated: lightPalette.cardBg,
  colorBorder: lightPalette.border,
  colorBorderSecondary: "#EEF1F6",
  colorText: lightPalette.text,
  colorTextSecondary: lightPalette.textSub,
  borderRadius: 10,
  fontSize: 14,
  wireframe: false,
};