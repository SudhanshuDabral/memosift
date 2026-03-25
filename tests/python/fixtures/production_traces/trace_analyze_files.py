# Production trace: "analyze these files" — 10-turn flowback analysis, $2.24 original cost.
# Source: debug_anlayze_these_files_2026-03-25.md
#
# This fixture contains a representative subset of the actual conversation messages
# converted to MemoSift's OpenAI-compatible format. Tool call arguments and results
# use real data from the production logs.
from __future__ import annotations

TRACE_MESSAGES: list[dict] = [
    # ── Turn 1: User query ──
    {
        "role": "user",
        "content": (
            "can you anlayze these files and tell me how we can comprehend this information"
        ),
    },
    # ── Turn 2: AI analysis with 8 tool calls (3 failures + retries) ──
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Analyzing files for comprehension**\n\n"
            "I need to analyze some files the user mentioned, but they aren't current "
            "uploads. I'll start by searching for these files. It seems I can use tools "
            "like `list_spreadsheet_files` and possibly an `analyze_data` specialist tool. "
            "Since there are only two spreadsheets, I can likely use direct spreadsheet "
            "tools for analysis.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t2_1",
                "type": "function",
                "function": {
                    "name": "list_spreadsheet_files",
                    "arguments": "{}",
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_1",
        "content": (
            '[{"fileRef":1,"fileName":"Formentera Roper North-FlowBack.xlsx",'
            '"fileType":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},'
            '{"fileRef":2,"fileName":"Formentera Roper South-Flowback.xlsx",'
            '"fileType":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}]'
        ),
    },
    # Tool call 2: analyze_spreadsheet — structural analysis (38KB original, summarized)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_2",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, os, json, glob, re, numpy as np\\n'
                        'files = sorted(glob.glob(\'/tmp/data/*\'))\\nsummary = {}\\n'
                        '# ... structural analysis code (3651 chars original) ...\\n'
                        'print(json.dumps({\'fileSummaries\': summary}, indent=2))",'
                        '"language":"python","fileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_2",
        "content": (
            '{"exitCode":0,"stdout":"{\\"fileSummaries\\": {\\"Formentera_Roper_North-FlowBack.xlsx\\": '
            '{\\"sheets\\": {\\"FESCO Test Results\\": {\\"effectiveShape\\": {\\"rows\\": 404, '
            '\\"cols\\": 241}, \\"columns\\": [\\"Real Time\\", \\"Test Time\\", \\"WHP\\", '
            '\\"Csg Press\\", \\"Gas Rate\\", \\"Oil Rate\\", \\"Water Rate\\", \\"Choke Size\\", '
            '\\"Total GOR\\", \\"Total WGR\\", \\"Cum Gas\\", \\"Cum Oil\\", \\"Cum Water\\", '
            '\\"Oil Cut\\", \\"Total GLR\\"]}}}}}}"}'
        ),
    },
    # Tool call 3: analyze_spreadsheet — FAILED (TypeError)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_3",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd\\ndf[c] = pd.to_numeric(df[c], errors=\'coerce\')",'
                        '"language":"python","fileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_3",
        "content": (
            '{"exitCode":1,"stderr":"Traceback (most recent call last):\\n'
            "  File \\\"<string>\\\", line 41, in <module>\\n"
            "TypeError: arg must be a list, tuple, 1-d array, or Series\\n\"}"
        ),
    },
    # Tool call 4: analyze_spreadsheet — FAILED (KeyError)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_4",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"df = df[df[\'Real Time\'].notna()]",'
                        '"language":"python","fileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_4",
        "content": (
            '{"exitCode":1,"stderr":"KeyError: \'Real Time\'\\n"}'
        ),
    },
    # Tool call 5: analyze_spreadsheet — SUCCESS (preview, 10KB original)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_5",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, glob, os, json\\n'
                        "files = sorted(glob.glob('/tmp/data/*'))\\n"
                        'print(json.dumps(out, indent=2))",'
                        '"language":"python","fileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_5",
        "content": (
            '{"exitCode":0,"stdout":"Headers row 4: Real Time, Test Time, WHP, Csg Press, '
            "Gas Rate, Oil Rate, Water Rate, Choke Size, Total GOR, Total WGR, Cum Gas, "
            'Cum Oil, Cum Water, Oil Cut, Total GLR\\nUnits row 5: mm/dd/yy HH:mm, Hours, '
            'psig, psig, Mcf/d, STB/d, STB/d, 16ths inch, Scf/STB, STB/MMcf, Mcf, STB, STB, %, Scf/STB"}'
        ),
    },
    # Tool call 6: analyze_spreadsheet — SUCCESS (comprehensive stats, 23KB original)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_6",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, glob, os, json, numpy as np\\n'
                        "# Full analysis with stats, peaks, samples\\n"
                        'print(json.dumps({\'results\': results, \'comparison\': comparison}, indent=2))",'
                        '"language":"python","fileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_6",
        "content": (
            '{"exitCode":0,"stdout":"{\\"results\\": {\\"Formentera_Roper_North-FlowBack.xlsx\\": '
            '{\\"metadata\\": {\\"well\\": \\"Formentera\\", \\"field\\": \\"Roper STX N731 H\\"}, '
            '\\"row_count\\": 404, \\"flowing_row_count\\": 394, '
            '\\"date_range\\": {\\"start\\": \\"2025-10-11T09:30:00\\", \\"end\\": \\"2025-10-28T05:00:00\\"}, '
            '\\"overall_stats\\": {\\"Gas Rate\\": {\\"avg\\": 1992.32, \\"max\\": 2573.0, \\"last\\": 2506.0}, '
            '\\"Oil Rate\\": {\\"avg\\": 1019.24, \\"max\\": 1478.4, \\"last\\": 1406.4}, '
            '\\"Water Rate\\": {\\"avg\\": 1648.67, \\"max\\": 2152.8, \\"last\\": 1828.8}, '
            '\\"WHP\\": {\\"avg\\": 1783.04, \\"max\\": 2405.0, \\"last\\": 1540.0}, '
            '\\"Csg Press\\": {\\"avg\\": 578.55, \\"max\\": 1100.0, \\"last\\": 450.0}, '
            '\\"Cum Gas\\": {\\"last\\": 32292.21}, \\"Cum Oil\\": {\\"last\\": 16732.47}, '
            '\\"Cum Water\\": {\\"last\\": 27398.6}, '
            '\\"Oil Cut\\": {\\"avg\\": 0.38, \\"last\\": 0.43}}}, '
            '\\"Formentera_Roper_South-Flowback.xlsx\\": '
            '{\\"metadata\\": {\\"well\\": \\"Formentera\\", \\"field\\": \\"Roper STX Unit S731 H\\"}, '
            '\\"row_count\\": 404, \\"flowing_row_count\\": 399, '
            '\\"overall_stats\\": {\\"Gas Rate\\": {\\"avg\\": 2261.28, \\"max\\": 3065.0, \\"last\\": 2866.0}, '
            '\\"Oil Rate\\": {\\"avg\\": 1154.67, \\"max\\": 1730.4, \\"last\\": 1723.2}, '
            '\\"Water Rate\\": {\\"avg\\": 1520.31, \\"max\\": 1946.4, \\"last\\": 1488.0}, '
            '\\"WHP\\": {\\"avg\\": 1951.38, \\"max\\": 3286.0, \\"last\\": 1660.0}, '
            '\\"Csg Press\\": {\\"avg\\": 779.95, \\"max\\": 1000.0, \\"last\\": 850.0}, '
            '\\"Cum Gas\\": {\\"last\\": 37593.75}, \\"Cum Oil\\": {\\"last\\": 19196.31}, '
            '\\"Cum Water\\": {\\"last\\": 25383.42}, '
            '\\"Oil Cut\\": {\\"avg\\": 0.42, \\"last\\": 0.54}}}}, '
            '\\"comparison\\": {\\"Gas Rate\\": {\\"+pct\\": 13.5}, \\"Oil Rate\\": {\\"+pct\\": 13.3}, '
            '\\"Water Rate\\": {\\"+pct\\": -7.8}, \\"WHP\\": {\\"+pct\\": 9.4}}}"}'
        ),
    },
    # Tool call 7: run_code_in_sandbox — percentage calculations
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_7",
                "type": "function",
                "function": {
                    "name": "run_code_in_sandbox",
                    "arguments": (
                        '{"code":"north = {\'avg_gas\':1992.32,\'avg_oil\':1019.24}\\n'
                        "south = {'avg_gas':2261.28,'avg_oil':1154.67}\\n"
                        "for k in ['avg_gas','avg_oil']:\\n"
                        "    pct = (south[k]/north[k]-1)*100\\n"
                        '    print(k, round(pct,1))","language":"python"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_7",
        "content": (
            '{"exitCode":0,"stdout":"avg_gas 13.5\\navg_oil 13.3\\navg_water -7.8\\n'
            "avg_whp 9.4\\nlast_cum_gas 16.4\\nlast_cum_oil 14.7\\nlast_cum_water -7.4\\n"
            'last_oil_cut 23.4\\navg_oil_cut 11.5\\nlast_gas 14.4\\nlast_oil 22.5\\nlast_water -18.6\\n"}'
        ),
    },
    # Tool call 8: run_code_in_sandbox — skipped
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_8",
                "type": "function",
                "function": {
                    "name": "run_code_in_sandbox",
                    "arguments": '{"code":"print(\'skip\')","language":"python"}',
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_8",
        "content": '{"exitCode":0,"stdout":"skip\\n"}',
    },
    # Turn 2 response: Full analysis with KPI tables
    {
        "role": "assistant",
        "content": (
            "## Summary\n\nYes — I analyzed both flowback spreadsheets. They are **hourly FESCO "
            "production test reports** for the **Roper North** and **Roper South** wells.\n\n"
            "## North vs South: key findings\n\n"
            "| Metric | North | South | Read-through |\n"
            "|---|---:|---:|---|\n"
            "| Avg gas rate, Mcf/d | **1,992** | **2,261** | South higher by ~**13.5%** |\n"
            "| Avg oil rate, STB/d | **1,019** | **1,155** | South higher by ~**13.3%** |\n"
            "| Avg water rate, STB/d | **1,649** | **1,520** | South lower by ~**7.8%** |\n"
            "| Avg WHP, psig | **1,783** | **1,951** | South maintains stronger pressure |\n"
            "| Final cumulative gas, Mcf | **32,292** | **37,594** | South higher by ~**16.4%** |\n"
            "| Final cumulative oil, STB | **16,732** | **19,196** | South higher by ~**14.7%** |\n"
            "| Final cumulative water, STB | **27,399** | **25,383** | South lower by ~**7.4%** |\n"
            "| Avg oil cut | **38.1%** | **42.4%** | South cleaner fluid mix |\n"
            "| Final oil cut | **43.5%** | **53.7%** | South ends materially stronger |\n\n"
            "### Peak observed values\n\n"
            "| Metric | North peak | South peak |\n"
            "|---|---:|---:|\n"
            "| Gas rate, Mcf/d | **2,573** | **3,065** |\n"
            "| Oil rate, STB/d | **1,478.4** | **1,730.4** |\n"
            "| Water rate, STB/d | **2,152.8** | **1,946.4** |\n"
            "| WHP, psig | **2,405** | **3,286** |\n\n"
            "**South appears to be the better-performing flowback well.**"
        ),
    },
    # ── Turn 6: Chart generation (Step 1) ──
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Considering chart creation**\n\n"
            "I need to generate charts using only prior results without re-reading raw data.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t6_1",
                "type": "function",
                "function": {
                    "name": "list_conversation_files",
                    "arguments": "{}",
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t6_1",
        "content": '{"files":[],"message":"No files found in this conversation."}',
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t6_2",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"chart","code":"import matplotlib.pyplot as plt\\n'
                        "import numpy as np\\n\\n"
                        "stages = ['Cleanup\\\\n(0-24 hr)', 'Transition\\\\n(Days 1-4)', "
                        "'Late-time\\\\n(Last 72 hr)']\\n"
                        "north = {'gas_rate': [492.89, 940.39, 2510.08], "
                        "'oil_rate': [332.23, 590.29, 1172.50], "
                        "'water_rate': [988.38, 946.11, 1870.22]}\\n"
                        "south = {'gas_rate': [414.32, 827.92, 2919.08], "
                        "'oil_rate': [375.80, 549.51, 1400.73], "
                        "'water_rate': [578.39, 1008.81, 1603.07]}\\n"
                        "# ... matplotlib chart generation code (~9752 chars total) ...\\n"
                        'fig.savefig(rate_path, dpi=300)\\nplt.close(fig)",'
                        '"outputFiles":["/tmp/output/rate_progression.png",'
                        '"/tmp/output/pressure_comparison.png",'
                        '"/tmp/output/cumulative_comparison.png",'
                        '"/tmp/output/oilcut_trend.png",'
                        '"/tmp/output/delta_summary.png"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t6_2",
        "content": (
            '{"exitCode":0,"stdout":"Generated chart files:\\n/tmp/output/rate_progression.png\\n",'
            '"uploadedFiles":[{"fileName":"rate_progression.png","fileSize":500764},'
            '{"fileName":"pressure_comparison.png","fileSize":362097},'
            '{"fileName":"cumulative_comparison.png","fileSize":272716},'
            '{"fileName":"oilcut_trend.png","fileSize":273364},'
            '{"fileName":"delta_summary.png","fileSize":246158}]}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Completed Step 1: presentation-quality comparison charts.\n\n"
            "Created 5 polished PNG visuals:\n"
            "1. Rate progression\n2. Pressure comparison\n"
            "3. Cumulative comparison\n4. Oil-cut trend\n5. Executive delta summary\n\n"
            "Key findings reinforced visually:\n"
            "- South gas: 2,919 vs 2,510 Mcf/d (late-time)\n"
            "- South oil: 1,401 vs 1,173 STB/d (late-time)\n"
            "- Avg water: 1,520 vs 1,649 STB/d\n"
            "- Avg oil cut: 0.42 vs 0.38\n"
            "- Final oil cut: 0.54 vs 0.43"
        ),
    },
    # ── Turn 7: PPTX generation (Step 2) — duplicate list_conversation_files + 2 failures ──
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Figuring out PPTX rendering**\n\n"
            "I need to render a PowerPoint presentation using helper functions.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t7_1",
                "type": "function",
                "function": {
                    "name": "list_conversation_files",
                    "arguments": "{}",
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t7_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"rate_progression.png","fileType":"image/png","fileSize":500764},'
            '{"fileRef":2,"fileName":"pressure_comparison.png","fileType":"image/png","fileSize":362097},'
            '{"fileRef":3,"fileName":"cumulative_comparison.png","fileType":"image/png","fileSize":272716},'
            '{"fileRef":4,"fileName":"oilcut_trend.png","fileType":"image/png","fileSize":273364},'
            '{"fileRef":5,"fileName":"delta_summary.png","fileType":"image/png","fileSize":246158}],"count":5}'
        ),
    },
    # Duplicate list_conversation_files call
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t7_2",
                "type": "function",
                "function": {
                    "name": "list_conversation_files",
                    "arguments": "{}",
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t7_2",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"rate_progression.png","fileType":"image/png","fileSize":500764},'
            '{"fileRef":2,"fileName":"pressure_comparison.png","fileType":"image/png","fileSize":362097},'
            '{"fileRef":3,"fileName":"cumulative_comparison.png","fileType":"image/png","fileSize":272716},'
            '{"fileRef":4,"fileName":"oilcut_trend.png","fileType":"image/png","fileSize":273364},'
            '{"fileRef":5,"fileName":"delta_summary.png","fileType":"image/png","fileSize":246158}],"count":5}'
        ),
    },
    # render_document FAILURE 1: AttributeError
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t7_3",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"pptx","code":"prs = create_branded_presentation(\\n'
                        "    title='North vs South Flowback Executive Summary')\\n"
                        "add_kpi_slide(prs, 'KPIs', kpis=[\\n"
                        "    ('Late Gas Rate Advantage', '+16.3%', 'South vs North')])\\n"
                        '# ... 6167 chars of PPTX generation code ...\\n",'
                        '"outputFiles":["/tmp/output/summary.pptx"],'
                        '"inputFileRefs":["1","2","3","4","5"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t7_3",
        "content": (
            '{"exitCode":1,"stderr":"AttributeError: \'tuple\' object has no attribute \'get\'"}'
        ),
    },
    # render_document FAILURE 2: TypeError
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t7_4",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"pptx","code":"prs = create_branded_presentation(\\n'
                        "    title='North vs South Flowback Executive Summary')\\n"
                        "add_kpi_slide(prs, 'KPIs', kpis=[\\n"
                        "    {'label': 'Late Gas Rate', 'value': '+16.3%'}])\\n"
                        "add_two_column_slide(prs, 'Interpretation', left=[...], right=[...])\\n"
                        '# ... 5938 chars of PPTX generation code ...\\n",'
                        '"outputFiles":["/tmp/output/summary.pptx"],'
                        '"inputFileRefs":["1","2","3","4","5"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t7_4",
        "content": (
            '{"exitCode":1,"stderr":"TypeError: add_two_column_slide() got an unexpected keyword argument \'left\'"}'
        ),
    },
    # render_document SUCCESS
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t7_5",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"pptx","code":"prs = create_branded_presentation(\\n'
                        "    title='North vs South Flowback Executive Summary')\\n"
                        "# Fixed: use positional args for two_column_slide\\n"
                        "add_two_column_slide(prs, 'Interpretation', left_points, right_points)\\n"
                        '# ... 5915 chars of PPTX generation code ...\\n",'
                        '"outputFiles":["/tmp/output/summary.pptx"],'
                        '"inputFileRefs":["1","2","3","4","5"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t7_5",
        "content": (
            '{"exitCode":0,"stdout":"Presentation saved: /tmp/output/summary.pptx\\n",'
            '"uploadedFiles":[{"fileName":"summary.pptx","fileSize":1494869}]}'
        ),
    },
    # Turn 7 response
    {
        "role": "assistant",
        "content": (
            "Done — I built the SolvxAI-branded PPTX executive summary.\n\n"
            "Key findings captured in the deck:\n"
            "- South outperforms North on:\n"
            "  - average gas rate: 2,261 vs 1,992 Mcf/d\n"
            "  - average oil rate: 1,155 vs 1,019 STB/d\n"
            "  - average WHP: 1,951 vs 1,783 psig\n"
            "  - cumulative gas: 37,594 vs 32,292 Mcf\n"
            "  - cumulative oil: 19,196 vs 16,732 STB\n"
            "  - oil cut: avg 0.42 vs 0.38, final 0.54 vs 0.43\n"
            "- South has lower water burden:\n"
            "  - average water rate: 1,520 vs 1,649 STB/d\n"
            "  - final cumulative water: 25,383 vs 27,399 STB"
        ),
    },
    # ── Turn 8: Full summary (no tool calls) ──
    {
        "role": "assistant",
        "content": (
            "## Summary\n\n"
            "I analyzed two **FESCO flowback test spreadsheets** for **Formentera's Roper North "
            "and Roper South wells**.\n\n"
            "## KPI comparison table\n\n"
            "| Metric | Unit | Roper North | Roper South | South vs North |\n"
            "|---|---:|---:|---:|---:|\n"
            "| Average Gas Rate | Mcf/d | **1,992.32** | **2,261.28** | **+13.5%** |\n"
            "| Average Oil Rate | STB/d | **1,019.24** | **1,154.67** | **+13.3%** |\n"
            "| Average Water Rate | STB/d | **1,648.67** | **1,520.31** | **-7.8%** |\n"
            "| Average WHP | psig | **1,783.04** | **1,951.38** | **+9.4%** |\n"
            "| Average Casing Pressure | psig | **578.55** | **779.95** | **+34.8%** |\n"
            "| Final Cum Gas | Mcf | **32,292.21** | **37,593.75** | **+16.4%** |\n"
            "| Final Cum Oil | STB | **16,732.47** | **19,196.31** | **+14.7%** |\n"
            "| Final Cum Water | STB | **27,398.60** | **25,383.42** | **-7.4%** |\n"
            "| Average Oil Cut | frac | **0.38** | **0.42** | **+10.5%** |\n"
            "| Final Oil Cut | frac | **0.43** | **0.54** | South materially stronger |\n\n"
            "## Bottom line\n\n"
            "**Roper South is the stronger flowback performer** versus **Roper North**."
        ),
    },
    # ── Turn 9: User asks to rebuild ──
    {
        "role": "user",
        "content": (
            "The last presentation that you built me got corrupted. Can you remake that "
            "presentation for me with better visuals because again the visuals that you "
            "created have an outline text? Recreate the visuals and recreate the presentation."
        ),
    },
    # ── Turn 10: Rebuild with 6 tool calls ──
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Processing user uploads**\n\n"
            "I need to recreate visuals and presentations due to corruption and outline "
            "text issues. I should start with listing conversation files.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t10_1",
                "type": "function",
                "function": {
                    "name": "list_conversation_files",
                    "arguments": "{}",
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t10_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"rate_progression.png","fileSize":500764},'
            '{"fileRef":2,"fileName":"pressure_comparison.png","fileSize":362097},'
            '{"fileRef":3,"fileName":"cumulative_comparison.png","fileSize":272716},'
            '{"fileRef":4,"fileName":"oilcut_trend.png","fileSize":273364},'
            '{"fileRef":5,"fileName":"delta_summary.png","fileSize":246158},'
            '{"fileRef":6,"fileName":"summary.pptx","fileSize":1494869}],"count":6}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t10_2",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, json, glob, os\\n'
                        "# Re-extract KPI stats from original Excel files\\n"
                        "summary = {}\\n# ... 2594 chars analysis code ...\\n"
                        'print(json.dumps(summary, indent=2))",'
                        '"language":"python","fileRefs":["Formentera Roper North-FlowBack.xlsx",'
                        '"Formentera Roper South-Flowback.xlsx"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t10_2",
        "content": (
            '{"exitCode":0,"stdout":"{\\"Roper North\\": {\\"rows\\": 404, '
            '\\"overall\\": {\\"Gas Rate\\": {\\"avg\\": 1992.32, \\"max\\": 2573.0, \\"last\\": 2506.0}, '
            '\\"Oil Rate\\": {\\"avg\\": 1019.24, \\"max\\": 1478.4, \\"last\\": 1406.4}}, '
            '\\"late72\\": {\\"Gas Rate\\": {\\"avg\\": 2510.08}, \\"Oil Rate\\": {\\"avg\\": 1172.5}}}, '
            '\\"Roper South\\": {\\"rows\\": 404, '
            '\\"overall\\": {\\"Gas Rate\\": {\\"avg\\": 2261.28, \\"max\\": 3065.0, \\"last\\": 2866.0}, '
            '\\"Oil Rate\\": {\\"avg\\": 1154.67, \\"max\\": 1730.4, \\"last\\": 1723.2}}, '
            '\\"late72\\": {\\"Gas Rate\\": {\\"avg\\": 2919.08}, \\"Oil Rate\\": {\\"avg\\": 1400.73}}}}"}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t10_3",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"chart","code":"import pandas as pd\\nimport numpy as np\\n'
                        "# Rebuild clean charts from raw data\\n"
                        '# ... 8278 chars of matplotlib code ...\\n",'
                        '"outputFiles":["/tmp/output/rebuilt_rate.png",'
                        '"/tmp/output/rebuilt_pressure.png",'
                        '"/tmp/output/rebuilt_cumulative.png",'
                        '"/tmp/output/rebuilt_oilcut.png",'
                        '"/tmp/output/rebuilt_delta.png"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t10_3",
        "content": (
            '{"exitCode":0,"stdout":"Created clean chart set",'
            '"uploadedFiles":[{"fileName":"rebuilt_rate.png","fileSize":392458},'
            '{"fileName":"rebuilt_pressure.png","fileSize":276650},'
            '{"fileName":"rebuilt_cumulative.png","fileSize":319383},'
            '{"fileName":"rebuilt_oilcut.png","fileSize":194130},'
            '{"fileName":"rebuilt_delta.png","fileSize":106126}]}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t10_4",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"pptx","code":"prs = create_branded_presentation(\\n'
                        "    title='North vs South Flowback Executive Summary (Rebuilt)')\\n"
                        "# ... 4064 chars of PPTX generation code ...\\n"
                        "save_presentation(prs, '/tmp/output/rebuilt_summary.pptx')\\n\","
                        '"outputFiles":["/tmp/output/rebuilt_summary.pptx"],'
                        '"inputFileRefs":["9","10","11","12","13"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t10_4",
        "content": (
            '{"exitCode":0,"stdout":"Presentation saved: /tmp/output/rebuilt_summary.pptx\\n",'
            '"uploadedFiles":[{"fileName":"rebuilt_summary.pptx","fileSize":1182868}]}'
        ),
    },
    # Turn 10 response
    {
        "role": "assistant",
        "content": (
            "## Summary\n\n"
            "I remade the visuals and rebuilt the PowerPoint with a **cleaner, simpler chart "
            "style**.\n\n"
            "The conclusions remain the same:\n"
            "- **Roper South is the stronger well overall**\n"
            "- South outperforms North on:\n"
            "  - **average gas rate**: **2,261 vs 1,992 Mcf/d**\n"
            "  - **average oil rate**: **1,155 vs 1,019 STB/d**\n"
            "  - **average WHP**: **1,965 vs 1,787 psig**\n"
            "  - **final cumulative gas**: **37,594 vs 32,292 Mcf**\n"
            "  - **final cumulative oil**: **19,196 vs 16,732 STB**\n"
            "  - **final oil cut**: **0.54 vs 0.43**\n"
            "- South also has **lower cumulative water**: **25,383 vs 27,399 STB**"
        ),
    },
]

# ── Critical facts that MUST survive compression ──
# These are the numerical KPIs that the agent's analysis depends on.
# If any are lost, the agent cannot produce correct summaries or presentations.
CRITICAL_FACTS: list[str] = [
    # Well identifiers
    "Roper North",
    "Roper South",
    "Pearsall",
    # North averages
    "1,992",      # North avg gas rate (Mcf/d)
    "1,019",      # North avg oil rate (STB/d)
    # South averages
    "2,261",      # South avg gas rate (Mcf/d)
    "1,154",      # South avg oil rate (STB/d) — or 1,155 rounded
    # Cumulative finals
    "32,292",     # North cum gas (Mcf)
    "37,593",     # South cum gas (Mcf) — or 37,594 rounded
    "16,732",     # North cum oil (STB)
    "19,196",     # South cum oil (STB)
    # Oil cut
    "0.43",       # North final oil cut
    "0.54",       # South final oil cut
    # Deltas
    "13.5",       # Gas rate delta %
    "13.3",       # Oil rate delta %
    # Peak values
    "2,573",      # North peak gas rate
    "3,065",      # South peak gas rate
    "2,405",      # North peak WHP
    "3,286",      # South peak WHP
    # Conclusion
    "South",      # South is the stronger well
]

# ── LLM Judge prompt for quality evaluation ──
LLM_JUDGE_PROMPT = """\
You are evaluating the quality of a compressed AI conversation. The original
conversation was a data analysis session where an AI agent analyzed two oil well
flowback spreadsheets (Roper North and Roper South) and produced comparison
charts and a PowerPoint presentation.

ORIGINAL CONVERSATION SUMMARY:
- Agent analyzed 404 hourly data points per well over 403.5 hours
- Compared gas rate, oil rate, water rate, WHP, casing pressure, oil cut
- South outperformed North on all key metrics
- Generated 5 comparison charts and an executive PowerPoint
- User requested a rebuild due to corrupted presentation

COMPRESSED CONVERSATION:
{compressed_text}

ANCHOR LEDGER (preserved facts):
{anchor_ledger}

Evaluate on these criteria (score 1-5 each):

1. **Fact Retention**: Are the key numerical KPIs preserved? (gas rates, oil rates,
   cumulative volumes, percentages, oil cut values)
2. **Decision Preservation**: Is the conclusion "South is the stronger well" clearly
   preserved with supporting evidence?
3. **Tool Call Coherence**: Can you tell what tools were called and what they produced?
   Are tool call/result pairs intact?
4. **Task Flow**: Can you follow the conversation flow? (analyze → chart → PPTX → rebuild)
5. **Actionability**: Could an agent continue this conversation from the compressed
   version without re-reading the spreadsheets?

Return JSON:
{{"fact_retention": <1-5>, "decision_preservation": <1-5>, "tool_coherence": <1-5>,
  "task_flow": <1-5>, "actionability": <1-5>, "overall": <1-5>,
  "missing_facts": ["list of important facts that were lost"],
  "assessment": "1-2 sentence summary"}}
"""
