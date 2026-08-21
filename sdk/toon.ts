// TypeScript implementation of TOON (Token-Oriented Object Notation)
export type JSObject = { [k: string]: any };

function escapeValue(val: any, delimiter = '|'): string {
  let s = val === null || val === undefined ? '' : String(val);
  s = s.replace(/\\/g, '\\\\');
  s = s.replace(/\n/g, '\\n');
  if (delimiter) {
    const re = new RegExp(escapeRegExp(delimiter), 'g');
    s = s.replace(re, '\\' + delimiter);
  }
  return s;
}

function unescapeValue(s: string): string {
  let res = '';
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '\\') {
      i++;
      if (i >= s.length) {
        res += '\\';
        break;
      }
      const nxt = s[i];
      if (nxt === 'n') res += '\n';
      else res += nxt;
    } else {
      res += c;
    }
  }
  return res;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function splitEscaped(line: string, delimiter = '|'): string[] {
  const parts: string[] = [];
  let cur = '';
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '\\') {
      if (i + 1 < line.length) {
        cur += '\\' + line[i + 1];
        i++;
      } else {
        cur += '\\';
      }
    } else if (delimiter && c === delimiter) {
      parts.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  parts.push(cur);
  return parts;
}

export function toToon(objects: JSObject[], fields?: string[], delimiter = '|', types?: string[], includeSchema = false): string {
  if (!objects || objects.length === 0) return '';
  if (!fields) fields = Object.keys(objects[0]);
  const header = fields.join(delimiter);
  let schemaLine = '';
  if (includeSchema) {
    if (!types) {
      types = fields.map((f) => {
        const v = objects[0][f];
        if (v === null || v === undefined) return 'str';
        if (typeof v === 'boolean') return 'bool';
        if (Number.isInteger(v)) return 'int';
        if (typeof v === 'number') return 'float';
        return 'str';
      });
    }
    schemaLine = types.join(delimiter);
  }
  const rows = objects.map((obj) => fields!.map((f) => escapeValue(obj[f], delimiter)).join(delimiter));
  const lines = [header];
  if (includeSchema) lines.push(schemaLine);
  return lines.concat(rows).join('\n');
}

export function fromToon(toon: string, delimiter = '|'): JSObject[] {
  if (!toon) return [];
  const lines = toon.split(/\r?\n/);
  const header = lines[0];
  const fields = header ? header.split(delimiter) : [];
  let types: string[] | undefined;
  let start = 1;
  if (lines.length > 1) {
    const cand = lines[1];
    const candParts = cand.split(delimiter);
    const common = new Set(['int', 'integer', 'float', 'double', 'number', 'str', 'string', 'bool', 'boolean']);
    if (candParts.every((p) => common.has(p.toLowerCase()))) {
      types = candParts;
      start = 2;
    }
  }
  const out: JSObject[] = [];
  for (let i = start; i < lines.length; i++) {
    const parts = splitEscaped(lines[i], delimiter).map((p) => unescapeValue(p));
    while (parts.length < fields.length) parts.push('');
    const obj: JSObject = {};
    for (let j = 0; j < fields.length; j++) {
      const typ = types && types[j] ? types[j] : undefined;
      obj[fields[j]] = castValue(parts[j], typ);
    }
    out.push(obj);
  }
  return out;
}

function castValue(s: string, typ?: string): any {
  if (!typ) return s;
  const t = typ.toLowerCase();
  if (s === '' && t !== 'str') return null;
  try {
    if (t === 'int' || t === 'integer') return parseInt(s, 10);
    if (t === 'float' || t === 'double' || t === 'number') return parseFloat(s);
    if (t === 'bool' || t === 'boolean') {
      const ls = s.toLowerCase();
      if (['true', '1', 'yes', 'y'].includes(ls)) return true;
      if (['false', '0', 'no', 'n'].includes(ls)) return false;
      return Boolean(s);
    }
    return s;
  } catch (e) {
    return s;
  }
}

export function chunkToon(objects: JSObject[], chunkSize = 100, fields?: string[], delimiter = '|', types?: string[], includeSchema = false) {
  if (chunkSize <= 0) throw new Error('chunkSize must be > 0');
  const total = Math.ceil(objects.length / chunkSize);
  const chunks: any[] = [];
  for (let i = 0; i < total; i++) {
    const start = i * chunkSize;
    const end = Math.min(start + chunkSize, objects.length);
    const subset = objects.slice(start, end);
    const toon = toToon(subset, fields, delimiter, types, includeSchema && i === 0);
    chunks.push({ chunkIndex: i, total, start, end, toon });
  }
  return chunks;
}

export function estimateTokens(text: string): number {
  if (!text) return 0;
  const wc = text.split(/\s+/).filter(Boolean).length;
  return Math.floor(wc * 1.3) + 1;
}

export function buildPromptForLLM(instruction: string, objects: JSObject[], fields?: string[], delimiter = '|', types?: string[], includeSchema = false) {
  const toon = toToon(objects, fields, delimiter, types, includeSchema);
  const prompt = `Data (TOON):\n${toon}\n\nInstructions:\n${instruction}`;
  return { prompt, tokensEstimate: estimateTokens(prompt) };
}
