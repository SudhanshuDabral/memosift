# Production trace: "understand these two" -- 9-turn Pearsall DCA analysis, $1.36 original cost.
# Source: debug_understand_these_two_2026-03-25.md
#
# This fixture contains a representative subset of the actual conversation messages
# converted to MemoSift's OpenAI-compatible format. Tool call arguments and results
# use real data from the production logs.
from __future__ import annotations

TRACE_MESSAGES: list[dict] = [
    # == Turn 1: User query with two spreadsheet attachments ==
    {
        "role": "user",
        "content": (
            "help me understand these two files and list the unique set of wells "
            "that has more than 1 year of production history"
        ),
    },
    # == Turn 2: AI analysis with 3 tool calls ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Analyzing spreadsheets for production data**\n\n"
            "I need to analyze some spreadsheets, starting by identifying sheet names "
            "and columns related to production history. I'll check if production history "
            "exceeds 365 days. I'll deduplicate by API_UWI to get unique wells.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t2_1",
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
        "tool_call_id": "tc_t2_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"Pearsall Well Production Transform '
            'to OIL TC_2023-06-16.xlsx","fileType":"application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet","fileSize":889276,"source":"user"},'
            '{"fileRef":2,"fileName":"Pearsall Well Production Transform to '
            'Condensate TC_2023-06-16.xlsx","fileType":"application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet","fileSize":1013340,"source":"user"}],'
            '"count":2}'
        ),
    },
    # Tool 2: analyze_spreadsheet -- structural analysis and well dedup
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
                        '{"code":"import pandas as pd, glob, os, json\\n'
                        "files = sorted(glob.glob('/tmp/data/*'))\\n"
                        "# Inspect sheets, columns, deduplicate wells by API_UWI\\n"
                        "# Calculate production history span per well\\n"
                        '... (2000 chars total)",'
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
            '{"exitCode":0,"stdout":"OIL workbook sheets: [\'Pearsall Well Production_2023-0\', '
            "'Modified Whitley Dubose', 'Economic Type Curves', 'For Pasting', 'Chart']\\n"
            "Condensate workbook sheets: ['Pearsall Well Production_2023-0', "
            "'Condensate Well', 'Economic Type Curves', 'Economic Type Curves (2)', "
            "'For Pasting', 'Chart']\\n"
            "Main production sheet: 3845 rows, 46 columns\\n"
            "Key columns: API_UWI, WellName, ProducingMonth, TotalProdMonths, "
            "ProducingDays, LiquidsProd_BBL, GasProd_MCF, WaterProd_BBL, "
            "CumLiquids_BBL, CumGas_MCF, CumWater_BBL\\n"
            "Unique wells total: 39\\n"
            "Wells with > 1 year history (>365 days): 37\\n"
            "Edge case: 42-163-33685 exactly 365 days, excluded under strict >365 rule\"}"
        ),
    },
    # Tool 3: run_code_in_sandbox -- verify well count
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t2_3",
                "type": "function",
                "function": {
                    "name": "run_code_in_sandbox",
                    "arguments": (
                        '{"code":"wells = [\\n'
                        "'42-013-34442','42-013-34565','42-013-34612',"
                        "'42-013-34619','42-013-34628','42-013-34629',"
                        "'42-013-34645','42-013-34751','42-013-34892',"
                        "'42-163-33447','42-163-33567','42-163-33574',"
                        "'42-163-33600','42-163-33608','42-163-33631',"
                        "'42-163-33637','42-163-33669','42-163-33670',"
                        "'42-163-33671','42-163-33672','42-163-33673',"
                        "'42-163-33683','42-163-33685','42-163-33700',"
                        "'42-163-33701','42-163-33709','42-163-33720',"
                        "'42-163-33730','42-163-33766','42-283-32289',"
                        "'42-283-32435','42-283-32441'\\n"
                        ']\\nprint(len(wells))","language":"python"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_3",
        "content": '{"exitCode":0,"stderr":"","stdout":"32\\n"}',
    },
    # Turn 2 response: Full analysis summary
    {
        "role": "assistant",
        "content": (
            "## Summary\n\n"
            "These two Excel workbooks are **type-curve / transformed production "
            "workbooks** built on what appears to be the **same underlying Pearsall "
            "monthly production dataset**. After combining and deduplicating the wells "
            "across both files, I found **39 unique wells total**, and **37 wells have "
            "more than 1 year of production history**.\n\n"
            "### What these two files contain\n\n"
            "Both files share a main sheet: **Pearsall Well Production_2023-0**\n"
            "- **3,845 rows**, **46 columns**\n"
            "- Monthly production by well / completion\n\n"
            "Key columns: API_UWI, WellName, ProducingMonth, TotalProdMonths, "
            "ProducingDays, LiquidsProd_BBL, GasProd_MCF, WaterProd_BBL, "
            "CumLiquids_BBL, CumGas_MCF, CumWater_BBL\n\n"
            "#### OIL workbook extra sheets:\n"
            "- Modified Whitley Dubose\n"
            "- Economic Type Curves\n"
            "- For Pasting / Chart\n\n"
            "#### Condensate workbook extra sheets:\n"
            "- Condensate Well\n"
            "- Economic Type Curves (1 and 2)\n"
            "- For Pasting / Chart\n\n"
            "### Wells with more than 1 year of production history\n\n"
            "**37 unique wells** with > 1 year history (strict >365 days rule)\n\n"
            "Representative API/UWI list includes:\n"
            "42-013-34442, 42-013-34565, 42-163-33447, 42-163-33600, "
            "42-163-33671, 42-283-32435, 42-283-32627, 42-163-33766, "
            "42-311-34312, 42-311-35043, etc.\n\n"
            "Edge case: **42-163-33685** -- exactly 365 days, excluded under >365 rule\n\n"
            "### Bottom line\n"
            "- These are **different analytical wrappers** around the same production dataset\n"
            "- **39 unique wells** total, **37** with > 1 year history"
        ),
    },
    # == Turn 3: User requests DCA analysis ==
    {
        "role": "user",
        "content": (
            "Can you do a DCA analysis on the best performing wells and create a "
            "time curve for me, after which create an executive-style document? "
            "I want to understand this play and see if it's viable or not."
        ),
    },
    # == Turn 4: AI asks clarification (2 tool calls) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Analyzing prior data files**\n\n"
            "I need to clarify criteria. Should 'best-performing' be based on "
            "cumulative liquids, BOE, or revenue? I also need to confirm format.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t4_1",
                "type": "function",
                "function": {
                    "name": "recall_memories",
                    "arguments": (
                        '{"query":"user preferences for play analysis reports, '
                        'document format, benchmarking Pearsall, DCA","limit":5}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t4_1",
        "content": (
            '{"memories":[{"category":"user_preference","content":"User prefers '
            'executive-style summaries and presentations.","confidence":0.5},'
            '{"category":"user_preference","content":"User is interested in '
            'benchmarking Pearsall versus Eagle Ford.","confidence":0.5},'
            '{"category":"user_preference","content":"User prefers ranked lists '
            'and tabular summaries.","confidence":0.5}],"count":5}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t4_2",
                "type": "function",
                "function": {
                    "name": "ask_clarification",
                    "arguments": (
                        '{"questions":[{"question":"What format for the executive '
                        'deliverable?","options":[{"label":"PPTX","value":"pptx"},'
                        '{"label":"DOCX","value":"docx"},{"label":"PDF","value":"pdf"}]},'
                        '{"question":"How to define best-performing wells?","options":['
                        '{"label":"Top by cumulative liquids/oil","value":"cum_liquids"},'
                        '{"label":"Top by first-12-month production","value":"first12"},'
                        '{"label":"Top by cumulative BOE","value":"cum_boe"},'
                        '{"label":"Balanced score","value":"balanced"}]},'
                        '{"question":"How many wells to include?","options":['
                        '{"label":"Top 5","value":"top5"},'
                        '{"label":"Top 10","value":"top10"},'
                        '{"label":"All 37 with >1yr","value":"all37"},'
                        '{"label":"Auto-select","value":"auto"}]},'
                        '{"question":"What viability lens?","options":['
                        '{"label":"Technical only","value":"technical"},'
                        '{"label":"Simple commercial screen","value":"commercial_simple"},'
                        '{"label":"Benchmark-style assessment","value":"benchmark"}]}]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t4_2",
        "content": '{"status":"waiting_for_user","questionCount":4}',
    },
    # Turn 4 response
    {
        "role": "assistant",
        "content": (
            "I found **39 unique Pearsall wells**, with **37 wells having more than "
            "1 year of production history**. Before I run the DCA, I need your "
            "choices on:\n\n"
            "1. **Output format** -- PPTX, DOCX, or PDF\n"
            "2. **How to define 'best-performing wells'** -- cumulative liquids/oil, "
            "first-12-month production, cumulative BOE, or balanced score\n"
            "3. **How many wells to include** -- top 5, top 10, all 37, or auto\n"
            "4. **What 'viability' means** -- technical only, simple commercial screen, "
            "or benchmark-style assessment"
        ),
    },
    # == Turn 5: User answers clarification ==
    {
        "role": "user",
        "content": (
            "Here are my answers to your clarification questions:\n\n"
            "What format for the executive deliverable?: docx\n"
            "How to define best-performing wells?: cum_boe\n"
            "How many wells to include?: top10\n"
            "What viability lens?: benchmark\n\n"
            "Please proceed with creating the plan based on these answers."
        ),
    },
    # == Turn 6: AI creates 2-step plan (1 tool call) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Creating a plan**\n\n"
            "Two steps: first analyze the production data, rank top 10 by cum BOE, "
            "fit DCA. Second, create a DOCX executive report.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t6_1",
                "type": "function",
                "function": {
                    "name": "create_plan",
                    "arguments": (
                        '{"reasoning":"DOCX executive deliverable, ranked by cumulative '
                        "BOE, top 10 wells, benchmark-style viability lens."
                        '","steps":[{"stepIndex":0,"title":"Analyze top 10 Pearsall wells '
                        'and build DCA/time curve","agentType":"spreadsheet_analysis"},'
                        '{"stepIndex":1,"title":"Create executive DOCX play assessment",'
                        '"agentType":"document_creation","dependsOn":[0]}]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t6_1",
        "content": (
            '{"status":"plan_created","stepCount":2,"reasoning":"DOCX executive '
            "deliverable, ranked by cumulative BOE, top 10 wells, benchmark-style "
            'viability lens."}'
        ),
    },
    # Turn 6 response
    {
        "role": "assistant",
        "content": (
            "## Proposed Plan\n\n"
            "1. **Analyze top 10 Pearsall wells and build DCA/time curve**\n"
            "   - Deduplicate wells by API/UWI\n"
            "   - Rank wells by cumulative BOE\n"
            "   - Perform decline curve analysis\n"
            "   - Build representative time curve\n\n"
            "2. **Create executive DOCX play assessment**\n"
            "   - Branded Word report with charts, tables, and conclusions"
        ),
    },
    # == Turn 7: AI executes analysis (7 tool calls to analyze_spreadsheet) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Planning execution steps**\n\n"
            "I need to compute core dataset metrics, rank top 10 by cum BOE, "
            "fit Arps decline models, and build P50 time curve.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t7_1",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, numpy as np, glob, os\\n'
                        "files = sorted(glob.glob('/tmp/data/*'))\\n"
                        "# Read both workbooks, deduplicate monthly production\\n"
                        "# Compute per-well metrics: cum BOE, first-12-mo BOE, peak rate\\n"
                        "# Rank and select top 10\\n"
                        '... (3000 chars total)",'
                        '"language":"python","fileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t7_1",
        "content": (
            '{"exitCode":0,"stdout":"Combined raw rows: 7690\\n'
            "After monthly dedup: 3845\\n"
            "Unique wells: 39\\n"
            "Eligible (>1yr): 37\\n\\n"
            "TOP 10 BY CUMULATIVE BOE:\\n"
            "1  42-163-33671  PALS RANCH 12H       362,783  152,487  120\\n"
            "2  42-163-33600  PALS RANCH 11H       228,999   94,472  127\\n"
            "3  42-283-32435  ZCW 1H               207,470   75,103  139\\n"
            "4  42-283-32627  WILSON C 1HP          200,841   75,557  133\\n"
            "5  42-163-33447  ALINE WILLIAMS B 1H   195,203   72,449  132\\n"
            "6  42-283-33601  CRAWFORD RANCH 12H    175,114   30,715  117\\n"
            "7  42-163-33766  WHITLEY-DUBOSE 1H     152,071   67,090  111\\n"
            "8  42-283-32626  ROBERT HINDES 1H      142,855   48,207  137\\n"
            "9  42-163-33700  WELDON SCHORP 101H    138,190   81,607  111\\n"
            "10 42-311-34312  RALPH EDWARDS E 1HP   123,345   46,720  103\"}"
        ),
    },
    # DCA fitting tool calls (summarized into one representative call)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t7_2",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, numpy as np\\n'
                        "# Arps decline curve analysis on top 10 wells\\n"
                        "# Fit exponential and hyperbolic, select best fit\\n"
                        "# Calculate 20-year screened EUR\\n"
                        "# Build P50 time curve milestones\\n"
                        '... (4000 chars total)",'
                        '"language":"python","fileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t7_2",
        "content": (
            '{"exitCode":0,"stdout":"DCA RESULTS:\\n'
            "PALS RANCH 12H: peak-12 636.7 BOE/d, cum 362.8 MBOE, "
            "model hyperbolic b=0.95, EUR20 413.8 MBOE\\n"
            "PALS RANCH 11H: peak-12 781.0 BOE/d, cum 229.0 MBOE, "
            "model hyperbolic b=0.50, EUR20 247.1 MBOE\\n"
            "ZCW 1H: peak-12 590.3 BOE/d, cum 207.5 MBOE, "
            "model hyperbolic b=0.75, EUR20 223.2 MBOE\\n"
            "WILSON C 1HP: peak-12 392.3 BOE/d, cum 200.8 MBOE, "
            "model hyperbolic b=0.80, EUR20 224.6 MBOE\\n"
            "ALINE WILLIAMS B 1H: peak-12 400.4 BOE/d, cum 195.2 MBOE, "
            "model hyperbolic b=0.10, EUR20 195.7 MBOE\\n"
            "CRAWFORD RANCH 12H: peak-12 168.7 BOE/d, cum 175.1 MBOE, "
            "model hyperbolic b=0.10, EUR20 180.3 MBOE\\n"
            "WHITLEY-DUBOSE 1H: peak-12 309.7 BOE/d, cum 152.1 MBOE, "
            "model hyperbolic b=0.65, EUR20 165.3 MBOE\\n"
            "ROBERT HINDES 1H: peak-12 286.1 BOE/d, cum 142.9 MBOE, "
            "model hyperbolic b=0.60, EUR20 154.1 MBOE\\n"
            "WELDON SCHORP 101H: peak-12 487.2 BOE/d, cum 138.2 MBOE, "
            "model hyperbolic b=0.10, EUR20 138.3 MBOE\\n"
            "RALPH EDWARDS E 1HP: peak-12 327.0 BOE/d, cum 123.3 MBOE, "
            "model hyperbolic b=0.10, EUR20 136.5 MBOE\\n\\n"
            "COHORT BENCHMARK:\\n"
            "Mean cum BOE: 192,687\\n"
            "Median cum BOE: 185,159\\n"
            "Mean first-12-mo BOE: 74,441\\n"
            "Median first-12-mo BOE: 73,776\\n"
            "Cum BOE CV: 0.358\\n"
            "First-12-mo BOE CV: 0.447\\n\\n"
            "TIME CURVE (P50):\\n"
            "Month  3: rate 235.5 BOE/d, norm 0.608, cum  16,682\\n"
            "Month  6: rate 184.2 BOE/d, norm 0.457, cum  34,709\\n"
            "Month 12: rate 132.9 BOE/d, norm 0.366, cum  73,776\\n"
            "Month 24: rate  73.4 BOE/d, norm 0.196, cum 104,840\\n"
            "Month 36: rate  46.5 BOE/d, norm 0.113, cum 128,520\\n"
            "Month 60: rate  23.0 BOE/d, norm 0.069, cum 160,272\\n"
            "Month 84: rate  16.7 BOE/d, norm 0.048, cum 173,168\\n"
            "Month120: rate  13.4 BOE/d, norm 0.029, cum 197,267 (6 wells)\\n\\n"
            "DECLINE RETENTION:\\n"
            "Month 12: 36%  Month 24: 20%  Month 60: 7%\\n\\n"
            "FLUID CHARACTER:\\n"
            "9 of 10 gas-weighted\\n"
            "WHITLEY-DUBOSE UNIT 1H most liquids-rich\\n\\n"
            "VIABILITY: Technically viable but heterogeneous. Best described as "
            "mixed-to-good in core pockets, not uniformly strong.\"}"
        ),
    },
    # Turn 7 response: analysis results
    {
        "role": "assistant",
        "content": (
            "## Top 10 Wells by Cumulative BOE\n\n"
            "| Rank | API/UWI | Well Name | Cum BOE | First 12 mo BOE | Months On |\n"
            "|---|---|---|---:|---:|---:|\n"
            "| 1 | 42-163-33671 | PALS RANCH 12H | 362,783 | 152,487 | 120 |\n"
            "| 2 | 42-163-33600 | PALS RANCH 11H | 228,999 | 94,472 | 127 |\n"
            "| 3 | 42-283-32435 | ZCW 1H | 207,470 | 75,103 | 139 |\n"
            "| 4 | 42-283-32627 | WILSON C 1HP | 200,841 | 75,557 | 133 |\n"
            "| 5 | 42-163-33447 | ALINE WILLIAMS UNIT B 1H | 195,203 | 72,449 | 132 |\n"
            "| 6 | 42-283-33601 | CRAWFORD RANCH 12H | 175,114 | 30,715 | 117 |\n"
            "| 7 | 42-163-33766 | WHITLEY-DUBOSE UNIT 1H | 152,071 | 67,090 | 111 |\n"
            "| 8 | 42-283-32626 | ROBERT HINDES 1H | 142,855 | 48,207 | 137 |\n"
            "| 9 | 42-163-33700 | WELDON SCHORP-SANDSPUR UNIT 101H | 138,190 | 81,607 | 111 |\n"
            "| 10 | 42-311-34312 | RALPH EDWARDS E 1HP | 123,345 | 46,720 | 103 |\n\n"
            "## Cohort benchmark\n"
            "- Mean cumulative BOE: **192,687**\n"
            "- Median cumulative BOE: **185,159**\n"
            "- Mean first-12-month BOE: **74,441**\n"
            "- Median first-12-month BOE: **73,776**\n"
            "- Cum BOE CV: **0.358** | First-12-mo BOE CV: **0.447**\n"
            "- 7 of 10 wells exceed 150 MBOE\n"
            "- 5 of 10 wells exceed 75 MBOE in first 12 months\n\n"
            "## DCA summary\n\n"
            "| Well | Model | b-Factor | EUR 20yr (BOE) | Peak-12 Rate (BOE/d) |\n"
            "|---|---|---:|---:|---:|\n"
            "| PALS RANCH 12H | Hyperbolic | 0.95 | 413,770 | 636.7 |\n"
            "| PALS RANCH 11H | Hyperbolic | 0.50 | 247,070 | 781.0 |\n"
            "| ZCW 1H | Hyperbolic | 0.75 | 223,218 | 590.3 |\n"
            "| WILSON C 1HP | Hyperbolic | 0.80 | 224,557 | 392.3 |\n"
            "| WHITLEY-DUBOSE UNIT 1H | Hyperbolic | 0.65 | 165,252 | 309.7 |\n\n"
            "## Representative time curve\n\n"
            "| Month | P50 Rate (BOE/d) | Normalized | P50 Cum BOE |\n"
            "|---:|---:|---:|---:|\n"
            "| 3 | 235.5 | 0.608 | 16,682 |\n"
            "| 6 | 184.2 | 0.457 | 34,709 |\n"
            "| 12 | 132.9 | 0.366 | 73,776 |\n"
            "| 24 | 73.4 | 0.196 | 104,840 |\n"
            "| 36 | 46.5 | 0.113 | 128,520 |\n"
            "| 60 | 23.0 | 0.069 | 160,272 |\n"
            "| 120* | 13.4 | 0.029 | 197,267 |\n\n"
            "*Month 120 has 6 wells still represented.\n\n"
            "## Decline retention\n"
            "- Month 12: **36%** | Month 24: **20%** | Month 60: **7%**\n\n"
            "## Fluid character\n"
            "- **9 of 10** top wells are **gas-weighted**\n"
            "- **WHITLEY-DUBOSE UNIT 1H** is the most liquids-rich\n\n"
            "## Benchmark viability conclusion\n"
            "**Technically viable but heterogeneous**: best described as "
            "'mixed-to-good in core pockets, not uniformly strong.'"
        ),
    },
    # == Turn 8: AI creates executive DOCX (2 tool calls) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Creating branded DOCX**\n\n"
            "I'll use render_document to create a DOCX with charts, tables, "
            "executive summary, DCA findings, and viability assessment.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t8_1",
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
        "tool_call_id": "tc_t8_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"Pearsall Well Production Transform '
            'to OIL TC_2023-06-16.xlsx","fileSize":889276,"source":"user"},'
            '{"fileRef":2,"fileName":"Pearsall Well Production Transform to '
            'Condensate TC_2023-06-16.xlsx","fileSize":1013340,"source":"user"}],'
            '"count":2}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t8_2",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"docx","code":"from docx import Document\\n'
                        "# Build branded executive report with 3 charts:\\n"
                        "# 1. Top 10 cum BOE bar chart\\n"
                        "# 2. Representative time curve (rate + normalized)\\n"
                        "# 3. Early-time vs long-term scatter\\n"
                        "# Tables: top 10 ranking, DCA summary, time-curve milestones\\n"
                        "# Sections: executive summary, scope/method, top-well ranking,\\n"
                        "# time curve, DCA findings, viability assessment, conclusions\\n"
                        '... (18237 chars total)",'
                        '"outputFiles":["/tmp/output/'
                        'Pearsall_Play_Assessment_Executive_Report.docx"],'
                        '"inputFileRefs":["1","2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t8_2",
        "content": (
            '{"exitCode":0,"stdout":"/tmp/output/'
            'Pearsall_Play_Assessment_Executive_Report.docx","uploadedFiles":'
            '[{"fileName":"Pearsall_Play_Assessment_Executive_Report.docx",'
            '"fileSize":323498}]}'
        ),
    },
    # Turn 8 response
    {
        "role": "assistant",
        "content": (
            "Done -- I created the executive DOCX report.\n\n"
            "### Deliverable\n"
            "- Pearsall_Play_Assessment_Executive_Report.docx\n\n"
            "### Key findings\n"
            "- Pearsall looks **technically viable in core pockets**, not uniformly strong\n"
            "- Top 10 wells average ~**192.7 MBOE** cumulative recovery\n"
            "- **7 of 10** wells exceed **150 MBOE**\n"
            "- Decline is **front-end loaded**: 36% at month 12, 20% at month 24, "
            "7% at month 60\n"
            "- Top cohort is **mostly gas-weighted**: commercial attractiveness depends "
            "on gas price, liquids yield, and product mix"
        ),
    },
    # == Turn 9: AI provides full summary (no tool calls) ==
    {
        "role": "assistant",
        "content": (
            "## Summary\n\n"
            "I analyzed the two uploaded Pearsall production workbooks, identified "
            "the best-performing wells using **top 10 wells by cumulative BOE**, ran "
            "a **screening-level decline curve analysis (DCA)**, built a "
            "**representative time curve**, and generated an **executive-style DOCX**.\n\n"
            "### Dataset summary\n\n"
            "| Metric | Value |\n"
            "|---|---:|\n"
            "| Combined raw monthly rows | 7,690 |\n"
            "| Unique monthly rows after dedup | 3,845 |\n"
            "| Unique wells | 39 |\n"
            "| Wells with >1 year history | 37 |\n\n"
            "### Top 10 wells by cumulative BOE\n\n"
            "| Rank | Well | Cum BOE | First 12-mo BOE | Months On |\n"
            "|---:|---|---:|---:|---:|\n"
            "| 1 | **PALS RANCH 12H** | 362,783 | 152,487 | 120 |\n"
            "| 2 | **PALS RANCH 11H** | 228,999 | 94,472 | 127 |\n"
            "| 3 | **ZCW 1H** | 207,470 | 75,103 | 139 |\n"
            "| 4 | **WILSON C 1HP** | 200,841 | 75,557 | 133 |\n"
            "| 5 | **ALINE WILLIAMS UNIT B 1H** | 195,203 | 72,449 | 132 |\n"
            "| 6 | **CRAWFORD RANCH 12H** | 175,114 | 30,715 | 117 |\n"
            "| 7 | **WHITLEY-DUBOSE UNIT 1H** | 152,071 | 67,090 | 111 |\n"
            "| 8 | **ROBERT HINDES 1H** | 142,855 | 48,207 | 137 |\n"
            "| 9 | **WELDON SCHORP-SANDSPUR UNIT 101H** | 138,190 | 81,607 | 111 |\n"
            "| 10 | **RALPH EDWARDS E 1HP** | 123,345 | 46,720 | 103 |\n\n"
            "### DCA highlights\n\n"
            "| Well | Model | b-Factor | EUR 20yr (BOE) | Last Rate (BOE/d) |\n"
            "|---|---|---:|---:|---:|\n"
            "| PALS RANCH 12H | Hyperbolic | 0.95 | 413,770 | 12.3 |\n"
            "| PALS RANCH 11H | Hyperbolic | 0.50 | 247,070 | 6.3 |\n"
            "| ZCW 1H | Hyperbolic | 0.75 | 223,218 | 6.0 |\n"
            "| WILSON C 1HP | Hyperbolic | 0.80 | 224,557 | 16.3 |\n"
            "| WHITLEY-DUBOSE UNIT 1H | Hyperbolic | 0.65 | 165,252 | 7.6 |\n\n"
            "### Viability conclusion\n\n"
            "**Overall:** Pearsall screens as **viable in selected/core areas**, but "
            "**not uniformly strong across the full well population**.\n\n"
            "**Why viable:** Best wells show credible cumulative recovery, durable "
            "production tail, 7 of 10 exceed 150 MBOE.\n\n"
            "**Why caution:** Variability is significant, decline is steep after "
            "early-time, cohort is mostly gas-weighted, strong results appear "
            "concentrated in better pockets.\n\n"
            "### Recommendations\n"
            "1. Add economics next (pricing, capex, opex)\n"
            "2. Benchmark against Eagle Ford\n"
            "3. Segment the play by area/operator/completion\n"
            "4. Build formal P10/P50/P90 type curves"
        ),
    },
]

# == Critical facts that MUST survive compression ==
# These are the numerical KPIs and identifiers that the agent's analysis depends on.
# If any are lost, the agent cannot produce correct summaries or presentations.
CRITICAL_FACTS: list[str] = [
    # Well identifiers
    "PALS RANCH 12H",
    "PALS RANCH 11H",
    "ZCW 1H",
    "WILSON C 1HP",
    "ALINE WILLIAMS UNIT B 1H",
    "CRAWFORD RANCH 12H",
    "WHITLEY-DUBOSE UNIT 1H",
    "ROBERT HINDES 1H",
    "WELDON SCHORP-SANDSPUR UNIT 101H",
    "RALPH EDWARDS E 1HP",
    # API/UWI identifiers
    "42-163-33671",     # PALS RANCH 12H
    "42-163-33600",     # PALS RANCH 11H
    "42-283-32435",     # ZCW 1H
    # Dataset metrics
    "7,690",            # Combined raw monthly rows
    "3,845",            # Unique monthly rows after dedup
    "39",               # Unique wells total
    "37",               # Wells with >1 year history
    # Cumulative BOE values (top wells)
    "362,783",          # PALS RANCH 12H cum BOE
    "228,999",          # PALS RANCH 11H cum BOE
    "207,470",          # ZCW 1H cum BOE
    "200,841",          # WILSON C 1HP cum BOE
    "195,203",          # ALINE WILLIAMS cum BOE
    "175,114",          # CRAWFORD RANCH 12H cum BOE
    "152,071",          # WHITLEY-DUBOSE cum BOE
    "142,855",          # ROBERT HINDES 1H cum BOE
    "138,190",          # WELDON SCHORP cum BOE
    "123,345",          # RALPH EDWARDS cum BOE
    # First 12-month BOE values
    "152,487",          # PALS RANCH 12H first-12-mo BOE
    "94,472",           # PALS RANCH 11H first-12-mo BOE
    "81,607",           # WELDON SCHORP first-12-mo BOE
    "75,557",           # WILSON C 1HP first-12-mo BOE
    "75,103",           # ZCW 1H first-12-mo BOE
    "72,449",           # ALINE WILLIAMS first-12-mo BOE
    "67,090",           # WHITLEY-DUBOSE first-12-mo BOE
    # DCA parameters
    "636.7",            # PALS RANCH 12H peak-12 rate (BOE/d)
    "781.0",            # PALS RANCH 11H peak-12 rate (BOE/d)
    "590.3",            # ZCW 1H peak-12 rate (BOE/d)
    "413,770",          # PALS RANCH 12H 20-yr EUR (BOE) -- or 413,769
    "247,070",          # PALS RANCH 11H 20-yr EUR (BOE)
    "223,218",          # ZCW 1H 20-yr EUR (BOE)
    "0.95",             # PALS RANCH 12H b-factor
    "0.50",             # PALS RANCH 11H b-factor
    "0.75",             # ZCW 1H b-factor
    "0.65",             # WHITLEY-DUBOSE b-factor
    # Cohort statistics
    "192,687",          # Mean cumulative BOE
    "185,159",          # Median cumulative BOE
    "74,441",           # Mean first-12-mo BOE
    "73,776",           # Median first-12-mo BOE
    "0.358",            # Cum BOE coefficient of variation
    "0.447",            # First-12-mo BOE coefficient of variation
    # Time curve milestones
    "235.5",            # P50 rate at month 3 (BOE/d)
    "132.9",            # P50 rate at month 12 (BOE/d)
    "73.4",             # P50 rate at month 24 (BOE/d)
    "197,267",          # P50 cum BOE at month 120
    # Decline retention
    "36",               # Rate retention at month 12 (%)
    "20",               # Rate retention at month 24 (%) -- or 19.6%
    "7",                # Rate retention at month 60 (%) -- or 6.9%
    # Fluid character
    "9",                # Gas-weighted wells out of top 10
    # Conclusion keywords
    "viable",
    "heterogeneous",
    "gas-weighted",
]

# == LLM Judge prompt for quality evaluation ==
LLM_JUDGE_PROMPT = """\
You are evaluating the quality of a compressed AI conversation. The original
conversation was a 9-turn data analysis session ($1.36 cost) where an AI agent
analyzed two Pearsall production workbooks, identified the top 10 wells by
cumulative BOE, performed screening-level decline curve analysis, built a
representative time curve, and generated an executive DOCX report with a
benchmark-style viability assessment.

ORIGINAL CONVERSATION SUMMARY:
- Agent analyzed two Pearsall production workbooks (OIL and Condensate type curves)
  sharing the same monthly production dataset of 3,845 unique monthly records
- 39 unique wells identified, 37 with >1 year of production history
- Top 10 wells ranked by cumulative BOE, led by PALS RANCH 12H (362,783 BOE)
- Screening-level DCA: all top 10 screened best with hyperbolic decline
- PALS RANCH 12H: peak-12 rate 636.7 BOE/d, 20-yr EUR 413,770 BOE, b=0.95
- Representative P50 time curve: 235.5 BOE/d at month 3, declining to
  13.4 BOE/d at month 120 with 197,267 cumulative BOE
- Decline retention: 36% at month 12, 20% at month 24, 7% at month 60
- 9 of 10 top wells are gas-weighted; WHITLEY-DUBOSE most liquids-rich
- Cohort mean cum BOE: 192,687; CV of 0.358 indicating meaningful dispersion
- Conclusion: Pearsall is technically viable in core pockets but heterogeneous

COMPRESSED CONVERSATION:
{compressed_text}

ANCHOR LEDGER (preserved facts):
{anchor_ledger}

Evaluate on these criteria (score 1-5 each):

1. **Fact Retention**: Are the key numerical KPIs preserved? (well rankings, cum BOE,
   DCA parameters, time curve milestones, decline retention, cohort stats)
2. **Decision Preservation**: Is the conclusion "technically viable but heterogeneous"
   preserved with supporting evidence? Is the gas-weighted nature noted?
3. **Tool Call Coherence**: Can you tell what tools were called and what they produced?
   Are tool call/result pairs intact? Is the clarification flow visible?
4. **Task Flow**: Can you follow the conversation flow? (understand files -> list wells ->
   ask clarification -> plan -> analyze + DCA -> time curve -> DOCX report -> summary)
5. **Actionability**: Could an agent continue this conversation (e.g., benchmark vs
   Eagle Ford, build P10/P50/P90 type curves) from the compressed version?

Return JSON:
{{"fact_retention": <1-5>, "decision_preservation": <1-5>, "tool_coherence": <1-5>,
  "task_flow": <1-5>, "actionability": <1-5>, "overall": <1-5>,
  "missing_facts": ["list of important facts that were lost"],
  "assessment": "1-2 sentence summary"}}
"""
