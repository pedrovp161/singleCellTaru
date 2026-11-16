// Converte o METODOS_preprocessamento.md (Markdown simples) em .docx.
// Uso (precisa do pacote npm docx): node raw/md2docx_metodos.js <entrada.md> <figura.png> <saida.docx>
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, ImageRun, LevelFormat, Footer, PageNumber,
} = require("docx");

const [, , src, figPath, out] = process.argv;
const lines = fs.readFileSync(src, "utf8").split("\n");
const FONT = "Arial";

// ── texto em linha: **negrito**, *itálico*, `código`, <sub>…</sub> ─────────
function runs(text, fmt = {}) {
  const out = [];
  const re = /(\*\*(.+?)\*\*|\*([^*]+?)\*|`([^`]+)`|<sub>(.*?)<\/sub>)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...fmt }));
    if (m[2] !== undefined) out.push(...runs(m[2], { ...fmt, bold: true }));
    else if (m[3] !== undefined) out.push(...runs(m[3], { ...fmt, italics: true }));
    else if (m[4] !== undefined) out.push(new TextRun({ text: m[4], font: "Consolas", size: 20, ...fmt }));
    else if (m[5] !== undefined) out.push(new TextRun({ text: m[5], font: FONT, subScript: true, ...fmt }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...fmt }));
  return out;
}

const body = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: /`[^`]{30,}`/.test(text) ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 }, ...opts,
});

// ── tabela ────────────────────────────────────────────────────────────────
function table(rows) {
  const cells = rows.filter(r => !/^\|\s*-/.test(r)).map(r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const widths = [2600, 2100, 2200, 2126];
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: widths.reduce((a, b) => a + b), type: WidthType.DXA },
    columnWidths: widths,
    rows: cells.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA }, borders,
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: "E8EEF6", color: "auto" } : undefined,
        margins: { top: 40, bottom: 40, left: 100, right: 100 },
        children: [new Paragraph({
          alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          children: runs(c, i === 0 ? { bold: true, size: 20 } : { size: 20 }),
        })],
      })),
    })),
  });
}

// ── figura ────────────────────────────────────────────────────────────────
function figure() {
  const w = 600, h = Math.round(w * 3040 / 2629);       // proporção do PNG
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, pageBreakBefore: true, spacing: { before: 0, after: 60 }, children: [
      new ImageRun({ type: "png", data: fs.readFileSync(figPath), transformation: { width: w, height: h },
                     altText: { title: "Figura 1", description: "Esquema do pré-processamento", name: "fig1" } }),
    ] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 240 }, children: [
      new TextRun({ text: "Figura 1. ", bold: true, font: FONT, size: 20 }),
      new TextRun({ text: "Esquema do pré-processamento a partir da matriz raw, com o número de células em cada etapa. " +
        "Caixas tracejadas: etapas que não alteram as células do dado final (avaliação da chamada de células e " +
        "análises posteriores). 14 bibliotecas de 7 indivíduos.", font: FONT, size: 20 }),
    ] }),
  ];
}

// ── percorre o Markdown ───────────────────────────────────────────────────
const children = [];
let para = [], list = null, tbl = [], section = "", figDone = false;

function flushPara() {
  if (para.length) children.push(body(para.join(" ")));
  para = [];
}
function flushList() {
  if (!list) return;
  for (const it of list) {
    const text = it.text.join(" ");
    if (section === "Referências") {
      children.push(new Paragraph({ children: runs(text), alignment: AlignmentType.LEFT,
        indent: { left: 360, hanging: 360 }, spacing: { after: 80 } }));
    } else {
      children.push(new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED,
        numbering: { reference: it.num ? "numeros" : "marcas", level: it.level }, spacing: { after: 60, line: 300 } }));
    }
  }
  list = null;
}
function flushTable() {
  if (tbl.length) { children.push(table(tbl)); children.push(new Paragraph({ spacing: { after: 120 }, children: [] })); }
  tbl = [];
}
function flushAll() { flushPara(); flushList(); flushTable(); }

for (const raw of lines) {
  const line = raw.replace(/\s+$/, "");
  if (line.startsWith(">")) continue;                    // nota interna do arquivo .md
  let m;
  if ((m = line.match(/^# (.+)/))) {
    flushAll();
    children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: runs(m[1]) }));
    children.push(new Paragraph({ spacing: { after: 240 }, children: [new TextRun({
      text: "Ressequenciamento de células T CD4⁺ na hanseníase — 14 bibliotecas, 7 indivíduos",
      font: FONT, size: 22, color: "555555", italics: true })] }));
    continue;
  }
  if ((m = line.match(/^(#{2,3}) (.+)/))) {
    flushAll();
    section = m[2];
    if (!figDone && m[1] === "##" && section.startsWith("Referências")) { children.push(...figure()); figDone = true; }
    children.push(new Paragraph({ heading: m[1] === "##" ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
                                  pageBreakBefore: section === "Referências",
                                  children: runs(m[2]) }));
    continue;
  }
  if (line.startsWith("|")) { flushPara(); flushList(); tbl.push(line); continue; }
  if (tbl.length) flushTable();
  if (!line.trim()) { flushPara(); continue; }
  if ((m = line.match(/^(\s*)(-|\d+\.) (.+)/))) {
    flushPara();
    list = list || [];
    list.push({ level: Math.min(1, Math.floor(m[1].length / 2)), num: m[2] !== "-", text: [m[3]] });
    continue;
  }
  if (list && /^\s+/.test(raw)) { list[list.length - 1].text.push(line.trim()); continue; }
  if (list) flushList();
  para.push(line.trim());
}
flushAll();

const heading = (size, before) => ({
  run: { font: FONT, size, bold: true, color: "1F3A5F" },
  paragraph: { spacing: { before, after: 120 }, keepNext: true },
});
const doc = new Document({
  creator: "pedrovp161", title: "Métodos — pré-processamento", language: "pt-BR",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
      title: { run: { font: FONT, size: 34, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 0, after: 60 }, keepNext: true } },
      heading1: heading(28, 300),
      heading2: heading(24, 200),
    },
  },
  numbering: { config: [
    { reference: "marcas", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 270 } } } } ] },
    { reference: "numeros", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 360 } } } } ] },
  ] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "777777" })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log("ok", out); });
