// pgn2shards.cpp — Lichess PGN → 动作列表分片（C++ 多进程高速构建器）
//
// 动作表与 stateseq/actions.py 逐字节一致（--dump-actions 供 Python 断言）。
// 过滤/元数据口径与 tools/stateseq_build_shards.py 一致（§4.1/D9）。
//
// 用法：
//   g++ -O3 -march=native -o tools/pgn2shards cpp/pgn2shards.cpp
//   tools/pgn2shards --pgn month.pgn --month 2026-08 --out data/shards --workers 16
//   tools/pgn2shards --dump-actions / --selfcheck / --verify --pgn x.pgn
//
// 分片格式 v2（gshards.py ShardReader 对应）：
//   shard-<month>-w<k>.actions.bin   : uint16 动作池（连续存放）
//   shard-<month>-w<k>.meta.bin      : 16B/局 {u16 n_plies,u8 tc,u8 result,
//                                        u8 elo_missing,u8 pad,f32 elo_mean,u32 local_idx,u32 pad}
//   shard-<month>-w<k>.elosample.bin : f32 Elo 抽样
// verify 模式输出（stdout，按文件序）：n_plies elo result fnv1a(actions)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;

// ---------------- 动作表（与 stateseq/actions.py 完全一致） ----------------
static const int QDIRS[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
static const int NJUMP[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};

struct ActionTables {
    std::array<int, 1 << 24> to_action;
    std::array<i8, 1936 * 3> from_tbl;
    int n = 0;
    ActionTables() { to_action.fill(-1); build(); }
    void add(int from, int to, int promo) {
        to_action[(from << 16) | (to << 8) | promo] = n;
        from_tbl[n * 3] = (i8)from; from_tbl[n * 3 + 1] = (i8)to; from_tbl[n * 3 + 2] = (i8)promo;
        n++;
    }
    void build() {
        for (int frm = 0; frm < 64; frm++) {
            int f0 = frm % 8, r0 = frm / 8;
            for (auto& d : QDIRS) {
                int f = f0 + d[0], r = r0 + d[1];
                while (f >= 0 && f < 8 && r >= 0 && r < 8) { add(frm, r * 8 + f, 0); f += d[0]; r += d[1]; }
            }
        }
        if (n != 1456) { std::cerr << "动作表错误: 后走法 " << n << "\n"; exit(1); }
        for (int frm = 0; frm < 64; frm++) {
            int f0 = frm % 8, r0 = frm / 8;
            for (auto& j : NJUMP) {
                int f = f0 + j[0], r = r0 + j[1];
                if (f >= 0 && f < 8 && r >= 0 && r < 8) add(frm, r * 8 + f, 0);
            }
        }
        if (n != 1792) { std::cerr << "动作表错误: 非升变 " << n << "\n"; exit(1); }
        int promo_from[16] = {48,49,50,51,52,53,54,55,8,9,10,11,12,13,14,15};
        int wdelta[3] = {8,7,9}, bdelta[3] = {-8,-9,-7};
        int promos[3] = {4,3,2};  // R,B,N
        for (int frm : promo_from) {
            int* deltas = frm >= 48 ? wdelta : bdelta;
            for (int di = 0; di < 3; di++) {
                int to = frm + deltas[di];
                bool straight = (deltas[di] == 8 || deltas[di] == -8);
                if (to < 0 || to >= 64 || std::abs(to % 8 - frm % 8) != (straight ? 0 : 1)) to = frm;
                for (int p : promos) add(frm, to, p);
            }
        }
        if (n != 1936) { std::cerr << "动作表错误: 总数 " << n << "\n"; exit(1); }
    }
    int move_to_action(int from, int to, int promo) const {
        int p = promo == 5 ? 0 : promo;
        return to_action[(from << 16) | (to << 8) | p];
    }
};

// ---------------- 棋盘与合法着生成 ----------------
struct Board {
    i8 sq[64];
    bool white;
    int castle, ep;
    void init() {
        for (auto& s : sq) s = 0;
        const int back[8] = {4,2,3,5,6,3,2,4};
        for (int f = 0; f < 8; f++) {
            sq[f] = (i8)back[f]; sq[8 + f] = 1; sq[48 + f] = -1; sq[56 + f] = (i8)-back[f];
        }
        white = true; castle = 0xF; ep = -1;
    }
    bool ray_attacked(int s, bool by_white, int begin, int end) const {
        int f = s % 8, r = s / 8;
        for (int d = begin; d < end; d++) {
            int ff = f + QDIRS[d][0], rr = r + QDIRS[d][1];
            while (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) {
                i8 p = sq[rr * 8 + ff];
                if (p) {
                    int a = std::abs(p);
                    bool rook_like = (begin == 0 && end == 4);
                    if ((rook_like && (a == 4 || a == 5)) || (!rook_like && (a == 3 || a == 5))) {
                        if ((p > 0) == by_white) return true;  // 找到同型攻击者
                    }
                    break;  // 被挡（或异色同型不攻击）：试下一方向
                }
                ff += QDIRS[d][0]; rr += QDIRS[d][1];
            }
        }
        return false;
    }
    bool is_attacked(int s, bool by_white) const {
        int f = s % 8, r = s / 8;
        for (auto& j : NJUMP) {
            int ff = f + j[0], rr = r + j[1];
            if (ff < 0 || ff >= 8 || rr < 0 || rr >= 8) continue;
            if (sq[rr * 8 + ff] == (by_white ? 2 : -2)) return true;
        }
        for (int df = -1; df <= 1; df++) for (int dr = -1; dr <= 1; dr++) {
            if (!df && !dr) continue;
            int ff = f + df, rr = r + dr;
            if (ff < 0 || ff >= 8 || rr < 0 || rr >= 8) continue;
            if (sq[rr * 8 + ff] == (by_white ? 6 : -6)) return true;
        }
        int pr = by_white ? r - 1 : r + 1;
        if (pr >= 0 && pr < 8) {
            for (int df : {-1, 1}) {
                int ff = f + df;
                if (ff < 0 || ff >= 8) continue;
                if (sq[pr * 8 + ff] == (by_white ? 1 : -1)) return true;
            }
        }
        return ray_attacked(s, by_white, 0, 4) || ray_attacked(s, by_white, 4, 8);
    }
    struct Mv { int from, to, promo; bool epm, cm; };
    void legal(std::vector<Mv>& out) const {
        out.clear();
        bool me = white;
        for (int s = 0; s < 64; s++) {
            i8 p = sq[s];
            if (!p || (p > 0) != me) continue;
            int a = std::abs(p), f = s % 8, r = s / 8;
            auto push = [&](int to, int promo = 0, bool epm = false, bool cm = false) {
                Board b = *this;
                b.apply(s, to, promo, epm, cm);
                int k = b.king_sq(me);
                if (k >= 0 && !b.is_attacked(k, !me)) out.push_back({s, to, promo, epm, cm});
            };
            if (a == 1) {
                int dir = me ? 8 : -8, start_r = me ? 1 : 6, promo_r = me ? 7 : 0;
                int r1 = r + (me ? 1 : -1);
                if (r1 >= 0 && r1 < 8) {
                    int to = s + dir;
                    if (!sq[to]) {
                        if (r1 == promo_r) { for (int pr : {5,4,3,2}) push(to, pr); }
                        else push(to);
                        if (r == start_r && !sq[s + 2 * dir]) push(s + 2 * dir);
                    }
                    for (int df : {-1, 1}) {
                        int ff = f + df;
                        if (ff < 0 || ff >= 8) continue;
                        int t2 = s + dir + df;
                        if (sq[t2] && (sq[t2] > 0) != me) {
                            if (r1 == promo_r) { for (int pr : {5,4,3,2}) push(t2, pr); }
                            else push(t2);
                        }
                        if (t2 == ep && sq[t2] == 0) push(t2, 0, true);
                    }
                }
            } else if (a == 2) {
                for (auto& j : NJUMP) {
                    int ff = f + j[0], rr = r + j[1];
                    if (ff < 0 || ff >= 8 || rr < 0 || rr >= 8) continue;
                    int to = rr * 8 + ff;
                    if (!sq[to] || (sq[to] > 0) != me) push(to);
                }
            } else {
                int b0 = a == 4 ? 0 : (a == 3 ? 4 : 0), b1 = a == 4 ? 4 : (a == 3 ? 8 : 8);
                for (int d = b0; d < b1; d++) {
                    int ff = f + QDIRS[d][0], rr = r + QDIRS[d][1];
                    while (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) {
                        int to = rr * 8 + ff;
                        if (!sq[to]) push(to);
                        else { if ((sq[to] > 0) != me) push(to); break; }
                        if (a == 6) break;  // 王只走一格
                        ff += QDIRS[d][0]; rr += QDIRS[d][1];
                    }
                }
                if (a == 6) {
                    if (me && s == 4) {
                        if ((castle & 1) && !sq[5] && !sq[6] && sq[7] == 4 &&
                            !is_attacked(4, false) && !is_attacked(5, false) && !is_attacked(6, false)) push(6, 0, false, true);
                        if ((castle & 2) && !sq[3] && !sq[2] && !sq[1] && sq[0] == 4 &&
                            !is_attacked(4, false) && !is_attacked(3, false) && !is_attacked(2, false)) push(2, 0, false, true);
                    } else if (!me && s == 60) {
                        if ((castle & 4) && !sq[61] && !sq[62] && sq[63] == -4 &&
                            !is_attacked(60, true) && !is_attacked(61, true) && !is_attacked(62, true)) push(62, 0, false, true);
                        if ((castle & 8) && !sq[59] && !sq[58] && !sq[57] && sq[56] == -4 &&
                            !is_attacked(60, true) && !is_attacked(59, true) && !is_attacked(58, true)) push(58, 0, false, true);
                    }
                }
            }
        }
    }
    int king_sq(bool side_white) const {
        i8 target = side_white ? 6 : -6;
        for (int i = 0; i < 64; i++) if (sq[i] == target) return i;
        return -1;
    }
    void apply(int from, int to, int promo, bool epm, bool cm) {
        i8 p = sq[from];
        bool me = p > 0;
        sq[from] = 0;
        if (epm) sq[me ? to - 8 : to + 8] = 0;
        sq[to] = promo ? (i8)(me ? promo : -promo) : p;
        if (cm) {
            if (to == 6) { sq[5] = sq[7]; sq[7] = 0; }
            if (to == 2) { sq[3] = sq[0]; sq[0] = 0; }
            if (to == 62) { sq[61] = sq[63]; sq[63] = 0; }
            if (to == 58) { sq[59] = sq[56]; sq[56] = 0; }
        }
        if (from == 4 || to == 4) castle &= ~3;
        if (from == 60 || to == 60) castle &= ~12;
        if (from == 0 || to == 0) castle &= ~2;
        if (from == 7 || to == 7) castle &= ~1;
        if (from == 56 || to == 56) castle &= ~8;
        if (from == 63 || to == 63) castle &= ~4;
        ep = -1;
        if (std::abs(p) == 1 && std::abs(to - from) == 16) ep = (from + to) / 2;
        white = !white;
    }
};

// ---------------- SAN 解析（解析即应用，返回动作 id） ----------------
struct San {
    static int parse(const std::string& tok_raw, Board& b, const ActionTables& T) {
        std::string tok = tok_raw;
        while (!tok.empty() && (tok.back() == '+' || tok.back() == '#' || tok.back() == '?' || tok.back() == '!')) tok.pop_back();
        if (tok.empty()) return -1;
        int ptype, to, promo = 0, ff = -1, rr = -1, fixed_from = -1;
        if (tok == "O-O" || tok == "0-0") { ptype = 6; to = b.white ? 62 % 64 : 62; to = b.white ? 6 : 62; fixed_from = b.white ? 4 : 60; }
        else if (tok == "O-O-O" || tok == "0-0-0") { ptype = 6; to = b.white ? 2 : 58; fixed_from = b.white ? 4 : 60; }
        else {
            size_t eq = tok.find('=');
            if (eq != std::string::npos) {
                if (eq + 1 >= tok.size()) return -1;
                char pl = tok[eq + 1];
                promo = pl == 'Q' ? 5 : pl == 'R' ? 4 : pl == 'B' ? 3 : pl == 'N' ? 2 : 0;
                if (!promo) return -1;
                tok.erase(eq, 2);
            }
            if (tok.size() < 2) return -1;
            char d0 = tok[tok.size() - 2], d1 = tok[tok.size() - 1];
            if (d0 < 'a' || d0 > 'h' || d1 < '1' || d1 > '8') return -1;
            to = (d1 - '1') * 8 + (d0 - 'a');
            std::string head = tok.substr(0, tok.size() - 2);
            size_t i = 0;
            ptype = 1;
            if (i < head.size() && head[i] >= 'A' && head[i] <= 'Z') {
                char pc = head[i++];
                ptype = pc == 'N' ? 2 : pc == 'B' ? 3 : pc == 'R' ? 4 : pc == 'Q' ? 5 : pc == 'K' ? 6 : 0;
                if (!ptype) return -1;
            }
            for (; i < head.size(); i++) {
                char c = head[i];
                if (c == 'x') continue;
                if (c >= 'a' && c <= 'h' && ff < 0) ff = c - 'a';
                else if (c >= '1' && c <= '8' && rr < 0) rr = c - '1';
                else return -1;
            }
        }
        static std::vector<Board::Mv> moves;
        b.legal(moves);
        int found = -1, count = 0;
        for (size_t k = 0; k < moves.size(); k++) {
            auto& m = moves[k];
            if (m.to != to) continue;
            if (std::abs(b.sq[m.from]) != ptype) continue;
            if (fixed_from >= 0 && m.from != fixed_from) continue;
            if (ff >= 0 && m.from % 8 != ff) continue;
            if (rr >= 0 && m.from / 8 != rr) continue;
            if (m.promo != promo) continue;
            found = (int)k; count++;
        }
        if (count != 1) return -1;
        auto& m = moves[found];
        int aid = T.move_to_action(m.from, m.to, m.promo);
        if (aid < 0) return -1;
        b.apply(m.from, m.to, m.promo, m.epm, m.cm);
        return aid;
    }
};

// ---------------- PGN 游标 ----------------
struct Cursor {
    const char* buf;
    size_t size, pos;
    Cursor(const char* b, size_t s, size_t p) : buf(b), size(s), pos(p) {}
    bool eof() const { return pos >= size; }
    char peek() const { return pos < size ? buf[pos] : '\0'; }
    char get() { return pos < size ? buf[pos++] : '\0'; }
    void skip_ws() { while (!eof() && (peek() == ' ' || peek() == '\t' || peek() == '\r' || peek() == '\n')) pos++; }
    std::string line() {
        skip_ws();
        size_t s0 = pos;
        while (!eof() && peek() != '\n') pos++;
        size_t len = pos - s0;
        if (!eof()) pos++;  // 消费换行
        return std::string(buf + s0, len);
    }
    std::string token() {
        skip_ws();
        size_t s0 = pos;
        while (!eof() && !strchr(" \t\r\n(){}", peek())) pos++;
        return std::string(buf + s0, pos - s0);
    }
    bool at_event() const {
        static const char pfx[] = "\n[Event ";
        return pos + sizeof(pfx) - 1 <= size && !memcmp(buf + pos, pfx, sizeof(pfx) - 1);
    }
};

static int tc_bucket(const std::string& tc) {
    if (tc.empty() || tc == "-" || tc == "?") return 6;
    if (tc.find('/') != std::string::npos) return 4;
    size_t plus = tc.find('+');
    try {
        long base = std::stol(plus == std::string::npos ? tc : tc.substr(0, plus));
        long inc = plus == std::string::npos ? 0 : std::stol(tc.substr(plus + 1));
        long secs = base + 40 * inc;
        if (secs < 180) return 0;
        if (secs < 480) return 1;
        if (secs < 1500) return 2;
        return 3;
    } catch (...) { return 6; }
}

struct GameOut {
    std::vector<u16> actions;
    int tc = 6, result = 1;
    bool elo_missing = true;
    float elo_mean = 1500.f;
};

// 解析一局（cur 位于 "[Event " 处）；返回 false = 该局不可用（坏棋/变体/无结果）
static bool parse_game(Cursor& cur, GameOut& out, const ActionTables& T) {
    std::string variant = "Standard", tc, result_str, welo, belo;
    while (true) {
        cur.skip_ws();
        if (cur.eof()) return false;
        if (cur.peek() != '[') break;
        std::string line = cur.line();   // 头逐行读（值可含空格）
        if (line.empty() || line[0] != '[') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string tv = line.substr(1, sp - 1);
        size_t v0 = line.find('"', sp), v1 = v0 == std::string::npos ? v0 : line.find('"', v0 + 1);
        std::string val = v0 == std::string::npos ? "" : line.substr(v0 + 1, v1 - v0 - 1);
        if (tv == "Variant") variant = val;
        else if (tv == "TimeControl") tc = val;
        else if (tv == "Result") result_str = val;
        else if (tv == "WhiteElo") welo = val;
        else if (tv == "BlackElo") belo = val;
    }
    int result = result_str == "1-0" ? 0 : result_str == "1/2-1/2" ? 1 : result_str == "0-1" ? 2 : -1;
    bool has_w = !welo.empty() && welo != "?", has_b = !belo.empty() && belo != "?";
    out.elo_missing = !(has_w || has_b);
    if (!out.elo_missing) {
        float e = 0; int n = 0;
        if (has_w) { e += std::stof(welo); n++; }
        if (has_b) { e += std::stof(belo); n++; }
        out.elo_mean = e / n;
    }
    out.tc = tc_bucket(tc);
    out.result = result < 0 ? 1 : result;
    if (variant != "Standard" || result < 0) {  // 跳过着法文本
        while (!cur.eof() && !cur.at_event()) cur.get();
        return false;
    }
    Board b; b.init();
    out.actions.clear();
    while (true) {
        cur.skip_ws();
        if (cur.eof() || cur.at_event()) break;
        char c = cur.peek();
        if (c == '{') { while (!cur.eof() && cur.get() != '}') {} continue; }
        if (c == '(') { int dep = 0; do { char ch = cur.get(); if (ch == '(') dep++; else if (ch == ')') dep--; } while (!cur.eof() && dep > 0); continue; }
        std::string tok = cur.token();
        if (tok.empty()) continue;
        if (tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*") break;
        if (tok[0] == '$') continue;
        // 走子序号前缀："1." / "1..." / "12.e4" / "1...e5"
        size_t dp = tok.find('.');
        if (dp != std::string::npos) {
            bool num = dp > 0;
            for (size_t k = 0; k < dp; k++) if (tok[k] < '0' || tok[k] > '9') { num = false; break; }
            if (num) {
                size_t k = dp;
                while (k < tok.size() && tok[k] == '.') k++;
                tok = tok.substr(k);
                if (tok.empty()) continue;
            }
        }
        if (tok.find_first_not_of("0123456789.") == std::string::npos) continue;
        int aid = San::parse(tok, b, T);
        if (aid < 0) {  // 坏着：跳到下一局
            while (!cur.eof() && !cur.at_event()) cur.get();
            return false;
        }
        out.actions.push_back((u16)aid);
    }
    return true;
}

static u64 fnv1a(const std::vector<u16>& v) {
    u64 h = 1469598103934665603ULL;
    for (u16 x : v) { h ^= x; h *= 1099511628211ULL; }
    return h;
}

// ---------------- worker ----------------
static void worker_run(int wid, const char* path, size_t begin, size_t end,
                       const std::string& month, const std::string& outdir,
                       bool verify_mode) {
    struct stat st; stat(path, &st);
    size_t fsize = st.st_size;
    end = std::min(end, fsize);
    FILE* fh = fopen(path, "rb");
    if (!fh) { std::cerr << "worker open fail\n"; exit(1); }
    char* buf = (char*)malloc(end - begin + 2);
    fseek(fh, begin, SEEK_SET);
    size_t got = fread(buf, 1, end - begin, fh);
    fclose(fh);
    buf[got] = '\0';
    // 起点对齐到下一局
    Cursor cur(buf, got, 0);
    if (begin > 0) {
        while (!cur.eof() && !cur.at_event()) cur.get();
        if (cur.eof()) exit(0);
    }
    static ActionTables T;
    std::string base = outdir + "/shard-" + month + "-w" + std::to_string(wid);
    FILE* fa = nullptr; FILE* fm = nullptr; FILE* fe = nullptr; FILE* fv = nullptr;
    if (verify_mode) {
        fv = fopen((base + ".verify.txt").c_str(), "w");
    } else {
        fa = fopen((base + ".actions.bin").c_str(), "wb");
        fm = fopen((base + ".meta.bin").c_str(), "wb");
        fe = fopen((base + ".elosample.bin").c_str(), "wb");
    }
    std::vector<u16> pool;
    std::vector<float> elosample;
    u32 kept = 0, skipped = 0;
    while (true) {
        cur.skip_ws();
        if (cur.eof()) break;
        if (cur.peek() != '[') { cur.get(); continue; }  // 对齐残留，逐字符跳过
        GameOut g;
        bool ok = parse_game(cur, g, T);
        if (!ok || g.actions.size() < 10) { skipped++; continue; }
        kept++;
        if (verify_mode) {
            fprintf(fv, "%zu %.1f %d %llu\n", g.actions.size(), g.elo_mean, g.result,
                    (unsigned long long)fnv1a(g.actions));
            continue;
        }
        for (u16 a : g.actions) pool.push_back(a);
        u8 meta[16];
        u16 np = (u16)g.actions.size();
        memcpy(meta, &np, 2);
        meta[2] = (u8)g.tc; meta[3] = (u8)g.result;
        meta[4] = g.elo_missing ? 1 : 0; meta[5] = 0;
        float em = g.elo_mean;
        memcpy(meta + 6, &em, 4);
        u32 li = kept - 1;
        memcpy(meta + 10, &li, 4);
        memset(meta + 14, 0, 2);
        fwrite(meta, 1, 16, fm);
        if (elosample.size() < 300000 && (kept % 40 == 0) && !g.elo_missing)
            elosample.push_back(g.elo_mean);
    }
    if (!verify_mode) {
        fwrite(pool.data(), 2, pool.size(), fa);
        fwrite(elosample.data(), 4, elosample.size(), fe);
        fclose(fa); fclose(fm); fclose(fe);
    } else fclose(fv);
    std::cerr << "worker " << wid << ": kept " << kept << " skipped " << skipped << "\n";
    free(buf);
}

// ---------------- 主流程 ----------------
int main(int argc, char** argv) {
    std::string pgn, month, outdir = "data/shards";
    int workers = 16;
    bool dump = false, selfcheck = false, verify = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--pgn") pgn = argv[++i];
        else if (a == "--month") month = argv[++i];
        else if (a == "--out") outdir = argv[++i];
        else if (a == "--workers") workers = atoi(argv[++i]);
        else if (a == "--dump-actions") dump = true;
        else if (a == "--selfcheck") selfcheck = true;
        else if (a == "--verify") verify = true;
    }
    static ActionTables T;
    if (dump) {
        for (int a = 0; a < 1936; a++)
            printf("%d %d %d\n", T.from_tbl[a*3], T.from_tbl[a*3+1], T.from_tbl[a*3+2]);
        return 0;
    }
    if (selfcheck) {
        // perft(4) 初始局面 = 197281
        struct R { long long nodes; };
        std::function<long long(Board&, int)> perft = [&](Board& b, int d) -> long long {
            if (d == 0) return 1;
            std::vector<Board::Mv> ms;   // 必须局部：递归内 legal() 会重入
            b.legal(ms);
            long long n = 0;
            for (auto& m : ms) { Board c = b; c.apply(m.from, m.to, m.promo, m.epm, m.cm); n += perft(c, d - 1); }
            return n;
        };
        Board b; b.init();
        long long n4 = perft(b, 4);
        std::cout << "perft(4) = " << n4 << (n4 == 197281 ? " OK" : " MISMATCH!") << "\n";
        // SAN 抽测（西班牙封闭真实棋线，含双易位/吃过路兵前置）
        const char* sans[] = {"e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7",
                              "Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3", "Be6"};
        Board g; g.init();
        bool ok = true;
        for (const char* s : sans) if (San::parse(std::string(s), g, T) < 0) { ok = false; std::cout << "SAN fail: " << s << "\n"; }
        std::cout << (ok ? "SAN OK" : "SAN FAIL") << "\n";
        return 0;
    }
    if (pgn.empty() || month.empty()) { std::cerr << "need --pgn and --month\n"; return 1; }
    struct stat st; stat(pgn.c_str(), &st);
    size_t size = st.st_size;
    mkdir(outdir.c_str(), 0777);
    // 找切分点（父进程）：每段对齐到 "\n[Event "
    std::vector<size_t> cuts(workers + 1);
    cuts[0] = 0; cuts[workers] = size;
    FILE* fh = fopen(pgn.c_str(), "rb");
    for (int w = 1; w < workers; w++) {
        size_t target = size_t((double)size * w / workers);
        fseek(fh, target, SEEK_SET);
        char window[1 << 20];
        size_t rd = fread(window, 1, sizeof(window), fh);
        std::string s(window, rd);
        size_t p = s.find("\n[Event ");
        cuts[w] = p == std::string::npos ? target : target + p;
    }
    fclose(fh);
    std::vector<pid_t> pids;
    for (int w = 0; w < workers; w++) {
        pid_t pid = fork();
        if (pid == 0) { worker_run(w, pgn.c_str(), cuts[w], cuts[w + 1], month, outdir, verify); exit(0); }
        pids.push_back(pid);
    }
    for (pid_t p : pids) waitpid(p, nullptr, 0);
    return 0;
}
