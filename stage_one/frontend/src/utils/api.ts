// 股票技术分析平台 - 前端 HTTP API 客户端
import {
  calculateSRLevels,
  calculateRangeStats,
  type KLinePoint,
  type SRLevel,
  type RangeStatPoint,
  type RangeStats,
} from './mockEngine';

export * from './symbols';
export {
  calculateSRLevels,
  calculateRangeStats,
  type KLinePoint,
  type SRLevel,
  type RangeStatPoint,
  type RangeStats,
};

export const BASE_URL = '/api';

export const SUPPORTED_PERIODS = ['5m', '15m', '1h', '4h', '1d', '1w', '1M', '3M', '6M', '1Y'] as const;
export type Period = (typeof SUPPORTED_PERIODS)[number];

export const PERIOD_LABELS: Record<Period, string> = {
  '5m': '5分',
  '15m': '15分',
  '1h': '1小时',
  '4h': '4小时',
  '1d': '日K',
  '1w': '周K',
  '1M': '月K',
  '3M': '季K',
  '6M': '半年K',
  '1Y': '年K',
};

export interface StockInfo {
  symbol: string;
  name: string;
  pinyin?: string;
}

export interface WatchGroup {
  id: string;
  name: string;
  stocks: string[];
}

export interface AnalysisResponse {
  sr_levels: SRLevel[];
  statistics: RangeStats;
}

export interface LeveragePairInfo {
  underlying: string;
  leverage: number;
  label: string;
}

export const LEVERAGE_ETF_MAP: Record<string, LeveragePairInfo> = {
  usNVDL: { underlying: 'usNVDA', leverage: 2, label: 'NVDA 2x做多' },
  usAMDL: { underlying: 'usAMD', leverage: 2, label: 'AMD 2x做多' },
  usMUU: { underlying: 'usMU', leverage: 2, label: 'MU 2x做多' },
  usTSMX: { underlying: 'usTSM', leverage: 2, label: 'TSM 2x做多' },
  usTSLL: { underlying: 'usTSLA', leverage: 2, label: 'TSLA 2x做多' },
  usTQQQ: { underlying: 'usQQQ', leverage: 3, label: 'QQQ 3x做多' },
  usSQQQ: { underlying: 'usQQQ', leverage: -3, label: 'QQQ 3x做空' },
  usSOXL: { underlying: 'usSOXX', leverage: 3, label: 'SOXX 3x做多' },
  usSOXS: { underlying: 'usSOXX', leverage: -3, label: 'SOXX 3x做空' },
};

/** 模糊检索股票列表 */
export async function searchStocks(keyword: string): Promise<StockInfo[]> {
  try {
    const url = keyword
      ? `${BASE_URL}/search?keyword=${encodeURIComponent(keyword)}`
      : `${BASE_URL}/search`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('网络请求异常，搜不到股票');
    return await response.json();
  } catch (error) {
    console.warn('Error searching stocks:', error);
    throw new Error('网络连接异常，搜不到该股票');
  }
}

/** 获取股票多周期 K 线数据 */
export async function getKLines(
  symbol: string,
  period: Period = '1d',
  full: boolean = false,
  fresh: boolean = false,
  signal?: AbortSignal
): Promise<KLinePoint[]> {
  const params = new URLSearchParams({ symbol, period });
  if (full) params.set('full', 'true');
  if (fresh) params.set('fresh', 'true');
  const response = await fetch(`${BASE_URL}/klines?${params.toString()}`, { cache: 'no-store', signal });
  if (!response.ok) {
    let detail = '服务器返回异常';
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) detail = errJson.detail;
    } catch {}
    throw new Error(`无法加载 K 线数据：${detail}`);
  }
  return await response.json();
}

/** 获取股票技术分析（支撑位、阻力位、区间涨跌统计），后端不可用时降级使用前端计算引擎 */
export async function getStockAnalysis(
  symbol: string,
  startDate: string,
  endDate: string,
  klineCache?: KLinePoint[],
  period: Period = '1d'
): Promise<AnalysisResponse | null> {
  try {
    const response = await fetch(`${BASE_URL}/analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, start_date: startDate, end_date: endDate, period }),
    });
    if (!response.ok) throw new Error('服务器端技术分析接口错误');
    return await response.json();
  } catch (error) {
    console.warn('Backend analysis failed, running frontend calculation:', error);
    const klines = klineCache && klineCache.length > 0 ? klineCache : await getKLines(symbol, period);
    if (!klines || klines.length === 0) throw new Error('无法提取 K 线行情进行分析');

    const sr_levels = calculateSRLevels(klines, 5, period);
    const statistics = calculateRangeStats(klines, startDate, endDate);

    return {
      sr_levels,
      statistics: statistics || {
        bars_count: 0,
        total_return_pct: 0,
        period_amplitude_pct: 0,
        avg_daily_range_pct: 0,
        win_rate_pct: 0,
        max_rise: { pct: 0, start: startDate.slice(0, 10), end: startDate.slice(0, 10) },
        max_fall: { pct: 0, start: startDate.slice(0, 10), end: startDate.slice(0, 10) },
      },
    };
  }
}

/** 自选分组持久化接口 */
export async function getWatchlist(): Promise<WatchGroup[]> {
  try {
    const response = await fetch(`${BASE_URL}/watchlist`, { method: 'GET' });
    if (!response.ok) throw new Error('load failed');
    const json = await response.json();
    if (Array.isArray(json.groups)) return json.groups as WatchGroup[];
    return [];
  } catch {
    return [];
  }
}

export async function saveWatchlist(groups: WatchGroup[]): Promise<void> {
  try {
    await fetch(`${BASE_URL}/watchlist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ groups }),
    });
  } catch {}
}

/** 执行价格行为交易策略 */
export async function runPAStrategy(
  symbol: string,
  period: string = '1d',
  mode: string = 'swing',
  klines?: KLinePoint[],
  signal?: AbortSignal
): Promise<any> {
  const body: any = { symbol, period, mode, draw_on_chart: true };
  if (klines && klines.length > 0) {
    body.klines = klines;
  }
  const response = await fetch(`${BASE_URL}/control/strategy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const err = await response.text();
    throw new Error(`策略执行失败: ${err}`);
  }
  return await response.json();
}

export interface RadarSignal {
  symbol: string;
  level: 'high' | 'medium' | 'none';
  tag: string;
  score: number;
  direction_mode: 'LONG' | 'SHORT' | 'WAIT' | string;
  passed_gates: boolean;
  rr_ratio: number;
  wave_pattern: string;
  dow_phase: string;
  dow_market_status: string;
  ideal_entry?: number | null;
}

export interface RadarScanResponse {
  period: string;
  updated_at: string;
  total: number;
  hit_count: number;
  signals: Record<string, RadarSignal>;
}

/** 批量拉取自选池雷达狙击信号 */
export async function fetchRadarScan(
  period: Period = '1d',
  refresh: boolean = false
): Promise<RadarScanResponse | null> {
  try {
    const params = new URLSearchParams({ period });
    if (refresh) params.set('refresh', 'true');
    const res = await fetch(`${BASE_URL}/radar/scan?${params.toString()}`, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export interface JournalEntry {
  id: number;
  symbol: string;
  period: string;
  side: 'LONG' | 'SHORT';
  status: 'OPEN' | 'CLOSED';
  entry_time: string;
  entry_price: number;
  shares: number;
  planned_stop?: number | null;
  planned_tp1?: number | null;
  planned_rr?: number | null;
  exit_time?: string | null;
  exit_price?: number | null;
  pnl_pct?: number | null;
  realized_r?: number | null;
  wave_pattern: string;
  dow_phase: string;
  dow_market_status: string;
  passed_gates: number;
  discipline_violation: string;
  notes: string;
  summary?: any;
  annotations?: any;
}

export interface JournalAttribution {
  total_closed: number;
  win_rate: number;
  avg_pnl_pct: number;
  avg_r: number;
  by_wave: Array<{ label: string; count: number; win_rate: number; avg_pnl_pct: number; avg_r: number }>;
  by_market_status: Array<{ label: string; count: number; win_rate: number; avg_pnl_pct: number; avg_r: number }>;
  by_gate_compliance: Array<{ label: string; count: number; win_rate: number; avg_pnl_pct: number; avg_r: number }>;
  violation_stats: Record<string, number>;
}

export async function createJournal(payload: {
  symbol: string; period: string; side: 'LONG' | 'SHORT';
  entry_time: string; entry_price: number; shares?: number; notes?: string;
  summary?: any; annotations?: any;
}): Promise<JournalEntry> {
  const res = await fetch(`${BASE_URL}/journal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('保存交易日志失败');
  const data = await res.json();
  return data.entry;
}

export async function closeJournal(id: number, exitPrice: number, exitTime: string, notes?: string): Promise<JournalEntry> {
  const res = await fetch(`${BASE_URL}/journal/${id}/close`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ exit_price: exitPrice, exit_time: exitTime, notes }),
  });
  if (!res.ok) throw new Error('平仓结算失败');
  const data = await res.json();
  return data.entry;
}

export async function fetchJournalList(symbol?: string, status?: string): Promise<JournalEntry[]> {
  const params = new URLSearchParams();
  if (symbol) params.set('symbol', symbol);
  if (status) params.set('status', status);
  const res = await fetch(`${BASE_URL}/journal?${params.toString()}`, { cache: 'no-store' });
  if (!res.ok) return [];
  const data = await res.json();
  return data.entries || [];
}

export async function fetchJournalDetail(id: number): Promise<JournalEntry | null> {
  const res = await fetch(`${BASE_URL}/journal/${id}`, { cache: 'no-store' });
  if (!res.ok) return null;
  return await res.json();
}

export async function deleteJournal(id: number): Promise<boolean> {
  const res = await fetch(`${BASE_URL}/journal/${id}`, { method: 'DELETE' });
  return res.ok;
}

export async function fetchJournalAttribution(): Promise<JournalAttribution | null> {
  const res = await fetch(`${BASE_URL}/journal/attribution`, { cache: 'no-store' });
  if (!res.ok) return null;
  return await res.json();
}
