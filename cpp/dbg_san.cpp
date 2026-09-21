#define main pgn2shards_main
#include "/home/jeefy/UniChess/SSM/cpp/pgn2shards.cpp"
#undef main
int main() {
    static ActionTables T;
    FILE* fh = fopen("/home/jeefy/UniChess/SSM/data/tmp_probe/bench.pgn", "rb");
    fseek(fh, 0, SEEK_END); long sz = ftell(fh); fseek(fh, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1); fread(buf, 1, sz, fh); buf[sz] = 0; fclose(fh);
    Cursor cur(buf, sz, 0);
    int fails = 0, shown = 0;
    while (true) {
        cur.skip_ws();
        if (cur.eof()) break;
        if (cur.peek() != '[') { cur.get(); continue; }
        std::string variant = "Standard", result_str;
        while (true) {
            cur.skip_ws();
            if (cur.eof() || cur.peek() != '[') break;
            std::string line = cur.line();
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string tv = line.substr(1, sp - 1);
            size_t v0 = line.find('"', sp);
            if (v0 == std::string::npos) continue;
            size_t v1 = line.find('"', v0 + 1);
            std::string val = line.substr(v0 + 1, v1 - v0 - 1);
            if (tv == "Variant") variant = val;
            if (tv == "Result") result_str = val;
        }
        if (variant != "Standard" || (result_str != "1-0" && result_str != "0-1" && result_str != "1/2-1/2")) {
            while (!cur.eof() && !cur.at_event()) cur.get();
            continue;
        }
        Board b; b.init();
        bool bad = false;
        std::string badtok;
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
            size_t dp = tok.find('.');
            if (dp != std::string::npos) {
                bool num = dp > 0;
                for (size_t k = 0; k < dp; k++) if (tok[k] < '0' || tok[k] > '9') { num = false; break; }
                if (num) { size_t k = dp; while (k < tok.size() && tok[k] == '.') k++; tok = tok.substr(k); if (tok.empty()) continue; }
            }
            if (tok.find_first_not_of("0123456789.") == std::string::npos) continue;
            if (San::parse(tok, b, T) < 0) { bad = true; badtok = tok; break; }
        }
        if (bad) {
            fails++;
            if (shown < 15) { printf("FAIL tok=<%s>\n", badtok.c_str()); shown++; }
            while (!cur.eof() && !cur.at_event()) cur.get();
        }
    }
    printf("total SAN fails: %d\n", fails);
    return 0;
}
