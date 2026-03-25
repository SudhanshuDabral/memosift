# Production trace: "examine these two" -- 20-turn Pearsall oil play analysis, $6.91 original cost.
# Source: debug_examine_these_two_2026-03-25.md
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
            "examine these two files and create a report on this play"
        ),
    },
    # == Turn 2: AI analysis via analyze_data tool ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Examining report requirements**\n\n"
            "I need to look over two conversation files and create a report "
            "on this play. The user mentioned 'this play,' likely referring "
            "to the files. I'll need to analyze the spreadsheets.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t2_1",
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "arguments": (
                        '{"input":"Use the conversation-uploaded files: '
                        "'Pearsall well vs 1st EF West Oil Metrics_2023-06-16.xlsx' "
                        "and 'Pearsall well vs 1st EF West Oil Metrics_2023-06-16.csv.xlsx'. "
                        "Determine whether they contain the same or different data, inspect "
                        "all sheets/tables, identify the play/area covered, and produce a "
                        "concise analytical report on the play... (838 chars total)"
                        '"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t2_1",
        "content": "(completed)",
    },
    # Turn 2 response: empty (analysis.json and play_summary.json attached)
    {
        "role": "assistant",
        "content": None,
    },
    # == Turn 3: User wants a document, not JSON ==
    {
        "role": "user",
        "content": "i need a document not json files",
    },
    # == Turn 4: AI creates DOCX report with 7 tool calls ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Generating report formats**\n\n"
            "I'm thinking about generating a DOCX report. I need to analyze "
            "the spreadsheet comprehensively and derive a summary of the play. "
            "I'll use render_document to create tables and charts.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t4_1",
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
        "tool_call_id": "tc_t4_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.xlsx","fileType":"application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet","fileSize":122563,"source":"user"},'
            '{"fileRef":2,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.csv.xlsx","fileType":"application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet","fileSize":197488,"source":"user"},'
            '{"fileRef":3,"fileName":"analysis.json","fileType":"application/json",'
            '"fileSize":56759,"source":"ai"},{"fileRef":4,"fileName":"play_summary.json",'
            '"fileType":"application/json","fileSize":21930,"source":"ai"}],"count":4}'
        ),
    },
    # Tool 2: analyze_spreadsheet -- structural analysis
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t4_2",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, numpy as np, glob, os\\n'
                        "path = [p for p in glob.glob('/tmp/data/*') if p.endswith('.xlsx')][0]\\n"
                        "df = pd.read_excel(path, sheet_name='BASE')\\n"
                        "pear = df[(df['Trajectory']=='HORIZONTAL') & "
                        "(df['ENVPlay']=='PEARSALL')].copy()\\n"
                        "ef = df[(df['Trajectory']=='HORIZONTAL') & "
                        "(df['ENVPlay']=='EAGLE FORD')].copy()\\n"
                        '... (1042 chars total)",'
                        '"language":"python","fileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t4_2",
        "content": (
            '{"exitCode":0,"injectedFiles":["/tmp/data/Pearsall_well_vs_1st_EF_'
            'West_Oil_Metrics_2023-06-16.csv.xlsx"],"stdout":"80 horizontal wells: '
            '26 Pearsall, 54 Eagle Ford\\nMedian GOR: Pearsall 9,692 vs EF 1,528\\n'
            'Median cum oil: Pearsall 32,346 vs EF 105,610\\n'
            'Median 12-mo oil: Pearsall 18,323 vs EF 30,136\\n'
            'Median months to peak: Pearsall 2.0 vs EF 4.5"}'
        ),
    },
    # Tool 3-7: Additional analysis, viability scoring, DOCX rendering (summarized)
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t4_3",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, numpy as np\\n'
                        "# Viability scoring with percentile-based ranking\\n"
                        "# Metrics: CumOil, First12MonthOil, PeakOil, WHLiquids, GOR\\n"
                        '... (3482 chars total)",'
                        '"language":"python","fileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t4_3",
        "content": (
            '{"exitCode":0,"stdout":"PEARSALL WELL COUNT 26\\n\\nTOP 10 BY VIABILITY SCORE\\n'
            "WHITLEY-DUBOSE UNIT 1H  ALVARADO 2014 LLC  FRIO  PRODUCING  95467.0  48201.0  "
            "8012.0  3557.0  63.0  98.4  93.6\\n"
            "PALS RANCH 12H  BLACKBRUSH O&G  FRIO  PRODUCING  91326.0  53880.0  "
            "9413.0  17834.0  25.0  83.7  93.9\\n"
            "ZCW 1H  CHEYENNE PETROLEUM  LA SALLE  PRODUCING  72193.0  31015.0  "
            "7139.0  11242.0  35.0  82.6  86.2\\n"
            "PALS RANCH 11H  BLACKBRUSH O&G  FRIO  PRODUCING  62815.0  38978.0  "
            "12156.0  15873.0  27.0  80.7  88.1\\n"
            "R.H. PICKENS ET AL B 101H  CRESCENT ENERGY  FRIO  INACTIVE PRODUCER  "
            "40757.0  25936.0  9345.0  4995.0  55.0  75.7  74.7\\n"
            "ALINE WILLIAMS UNIT B 1H  CHEYENNE PETROLEUM  FRIO  PRODUCING  "
            "63964.0  28793.0  4888.0  12310.0  33.0  74.2  76.2\\n"
            "WELDON SCHORP-SANDSPUR UNIT 101H  CRESCENT ENERGY  FRIO  PRODUCING  "
            "42898.0  28433.0  6995.0  13327.0  31.0  72.4  74.6\\n"
            "\\nTIER COUNTS {'Tier 2': 12, 'Tier 3': 7, 'Tier 1': 7}\\n"
            "\\nCOUNTY SUMMARY\\nATASCOSA  6  57.0  53.2  32795.5  5630.0\\n"
            "FRIO  15  55.5  54.6  30566.5  9339.0\\n"
            "LA SALLE  4  50.7  54.6  46643.5  18450.5\\n"
            "MCMULLEN  1  20.6  25.8  19751.0  31468.0\\n"
            "\\nOPERATOR SUMMARY\\nCRESCENT ENERGY COMPANY  12  46.4  39.8  25929.5\\n"
            "CHEYENNE PETROLEUM CORPORATION  3  78.4  81.2  68078.5\\n"
            "BLACKBRUSH O&G  2  82.2  91.0  77070.5\\n"
            'ALVARADO 2014 LLC  1  98.4  93.6  95467.0"}'
        ),
    },
    # Tool 4: render DOCX report
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t4_4",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"docx","code":"from docx import Document\\n'
                        "# Build branded DOCX report with tables and charts\\n"
                        '... (8500 chars total)",'
                        '"outputFiles":["/tmp/output/Pearsall_Play_Benchmark_Report.docx"],'
                        '"inputFileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t4_4",
        "content": (
            '{"exitCode":0,"stdout":"Report saved","uploadedFiles":[{"fileName":'
            '"Pearsall_Play_Benchmark_Report.docx","fileSize":310465}]}'
        ),
    },
    # Turn 4 response: Full DOCX report summary
    {
        "role": "assistant",
        "content": (
            "## Pearsall Play Benchmark Report\n\n"
            "I analyzed the two uploaded datasets covering **80 horizontal wells**:\n"
            "- **26 Pearsall**\n"
            "- **54 Eagle Ford benchmark wells**\n\n"
            "### Key Findings\n\n"
            "| Metric | Pearsall | Eagle Ford West |\n"
            "|---|---:|---:|\n"
            "| Well count | **26** | **54** |\n"
            "| Median cumulative oil, bbl | **32,346** | **105,610** |\n"
            "| Median first-12-mo oil, bbl | **18,323** | **30,136** |\n"
            "| Median GOR, scf/bbl | **9,692** | **1,528** |\n"
            "| Median liquids cut, % | **38** | **80** |\n"
            "| Median months to peak | **2.0** | **4.5** |\n\n"
            "- **Pearsall is much gassier** than the Eagle Ford benchmark\n"
            "- **Oil performance is weaker** in Pearsall on a liquids basis\n"
            "- Pearsall wells tend to **peak earlier**\n\n"
            "**Download:** Pearsall_Play_Benchmark_Report.docx"
        ),
    },
    # == Turn 5: User asks which wells are more viable ==
    {
        "role": "user",
        "content": "can you tell me which wells are more viable and better",
    },
    # == Turn 6: AI viability analysis with 2 tool calls ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Analyzing well viability**\n\n"
            "I need to define what 'viable and better' means in operational terms -- "
            "likely focusing on higher oil production, decent gas-to-oil equivalent, "
            "ongoing production, and strong first-year and cumulative results.\n```"
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
        "content": (
            '{"files":[{"fileRef":1,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.xlsx","fileSize":122563,"source":"user"},'
            '{"fileRef":2,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.csv.xlsx","fileSize":197488,"source":"user"},'
            '{"fileRef":5,"fileName":"Pearsall_Play_Benchmark_Report.docx",'
            '"fileSize":310465,"source":"ai"}],"count":5}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t6_2",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, numpy as np, glob, os\\n'
                        "path = [p for p in glob.glob('/tmp/data/*') if p.endswith('.xlsx')][0]\\n"
                        "df = pd.read_excel(path, sheet_name='BASE')\\n"
                        "pear = df[(df['Trajectory']=='HORIZONTAL') & "
                        "(df['ENVPlay']=='PEARSALL')].copy()\\n"
                        "# Viability scoring with percentile-based ranking\\n"
                        '... (3482 chars total)",'
                        '"language":"python","fileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t6_2",
        "content": (
            '{"exitCode":0,"stdout":"PEARSALL WELL COUNT 26\\n\\nTOP 10 BY VIABILITY SCORE\\n'
            "WHITLEY-DUBOSE UNIT 1H  98.4  93.6\\n"
            "PALS RANCH 12H  83.7  93.9\\n"
            "ZCW 1H  82.6  86.2\\n"
            "PALS RANCH 11H  80.7  88.1\\n"
            "R.H. PICKENS ET AL B 101H  75.7  74.7\\n"
            "ALINE WILLIAMS UNIT B 1H  74.2  76.2\\n"
            "WELDON SCHORP-SANDSPUR UNIT 101H  72.4  74.6\\n\\n"
            "BOTTOM 8:\\nALONZO PEELER A 1H  -4.0  5.2\\n"
            "ROBERT HINDES 1H  19.8  20.6\\n"
            "CHILIPITIN LTD. 103H  20.0  12.8\\n"
            "RALPH EDWARDS E 1HP  20.6  25.8\\n\\n"
            "COUNTY SUMMARY\\nATASCOSA  57.0  FRIO  55.5  LA SALLE  50.7  MCMULLEN  20.6\\n"
            "OPERATOR SUMMARY\\nCRESCENT ENERGY  12 wells  46.4\\n"
            "CHEYENNE PETROLEUM  3 wells  78.4\\n"
            'BLACKBRUSH O&G  2 wells  82.2"}'
        ),
    },
    # Turn 6 response: viability ranking
    {
        "role": "assistant",
        "content": (
            "## Best Pearsall Wells by Overall Viability\n\n"
            "| Rank | Well | Operator | County | Status | Cum Oil, bbl | "
            "12-mo Oil, bbl | Peak Oil, bbl/d | GOR, scf/bbl | Liquids % |\n"
            "|---|---|---|---|---|---:|---:|---:|---:|---:|\n"
            "| 1 | **WHITLEY-DUBOSE UNIT 1H** | ALVARADO 2014 LLC | FRIO | Producing | "
            "**95,467** | **48,201** | 8,012 | **3,557** | **63** |\n"
            "| 2 | **PALS RANCH 12H** | BLACKBRUSH O&G | FRIO | Producing | 91,326 | "
            "**53,880** | 9,413 | 17,834 | 25 |\n"
            "| 3 | **ZCW 1H** | CHEYENNE PETROLEUM | LA SALLE | Producing | 72,193 | "
            "31,015 | 7,139 | 11,242 | 35 |\n"
            "| 4 | **PALS RANCH 11H** | BLACKBRUSH O&G | FRIO | Producing | 62,815 | "
            "38,978 | **12,156** | 15,873 | 27 |\n"
            "| 5 | **R.H. PICKENS, ET AL 'B' 101H** | CRESCENT ENERGY | FRIO | "
            "Inactive Producer | 40,757 | 25,936 | 9,345 | **4,995** | **55** |\n"
            "| 6 | **ALINE WILLIAMS UNIT B 1H** | CHEYENNE PETROLEUM | FRIO | "
            "Producing | 63,964 | 28,793 | 4,888 | 12,310 | 33 |\n"
            "| 7 | **WELDON SCHORP-SANDSPUR UNIT 101H** | CRESCENT ENERGY | FRIO | "
            "Producing | 42,898 | 28,433 | 6,995 | 13,327 | 31 |\n\n"
            "### Most viable overall: **WHITLEY-DUBOSE UNIT 1H**\n"
            "- Best balance of high oil, good first-year performance, low GOR, "
            "high liquids cut, and still producing\n\n"
            "### Strongest pure producer: **PALS RANCH 12H**\n"
            "- Top oil and strongest early oil producer but very gassy\n\n"
            "### Best counties: Atascosa (median viability), Frio (deepest inventory)\n"
            "### Best operators: BlackBrush O&G, Cheyenne Petroleum, Alvarado 2014 LLC\n"
            "### Weakest: ALONZO PEELER A 1H (EOG), ROBERT HINDES 1H (EOG)"
        ),
    },
    # == Turn 7: User asks for executive PPT ==
    {
        "role": "user",
        "content": "now create a nice ppt with all our findings executive style",
    },
    # == Turn 8: AI creates executive PPTX (4 tool calls, 2 failures + retry) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Clarifying presentation needs**\n\n"
            "The user wants an executive-style PowerPoint. I'll create a 6-8 slide "
            "deck with charts, KPI tables, and management implications.\n```"
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
            '{"files":[{"fileRef":1,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.xlsx","fileSize":122563,"source":"user"},'
            '{"fileRef":2,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.csv.xlsx","fileSize":197488,"source":"user"},'
            '{"fileRef":5,"fileName":"Pearsall_Play_Benchmark_Report.docx",'
            '"fileSize":310465,"source":"ai"}],"count":5}'
        ),
    },
    # render_document FAILURE 1: AttributeError on kpi tuples
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
                        '{"format":"pptx","code":"import pandas as pd, numpy as np\\n'
                        "# Build branded PPTX with charts and KPI slides\\n"
                        "add_kpi_slide(prs, 'Headline KPIs', kpis=[\\n"
                        "    ('Pearsall wells', str(len(pear)))])\\n"
                        '... (11032 chars total)",'
                        '"outputFiles":["/tmp/output/Pearsall_Play_Executive_Review.pptx"],'
                        '"inputFileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t8_2",
        "content": (
            '{"exitCode":1,"stderr":"AttributeError: \'tuple\' object has no '
            "attribute 'get'\"}"
        ),
    },
    # render_document FAILURE 2: TypeError on keyword args
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t8_3",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"pptx","code":"import pandas as pd, numpy as np\\n'
                        "add_two_column_slide(prs, 'Where the Better Wells Are',\\n"
                        "    left=[...], right=[...])\\n"
                        '... (9965 chars total)",'
                        '"outputFiles":["/tmp/output/Pearsall_Play_Executive_Review.pptx"],'
                        '"inputFileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t8_3",
        "content": (
            '{"exitCode":1,"stderr":"TypeError: add_two_column_slide() got an '
            "unexpected keyword argument 'left'\"}"
        ),
    },
    # render_document SUCCESS
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t8_4",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"pptx","code":"import pandas as pd, numpy as np\\n'
                        "# Fixed: use dict for kpis, positional args for two_column_slide\\n"
                        "add_kpi_slide(prs, 'Headline KPIs', kpis=[\\n"
                        "    {'label':'Pearsall wells','value':str(len(pear))}])\\n"
                        "add_two_column_slide(prs, 'Where the Better Wells Are',\\n"
                        "    left_points, right_points)\\n"
                        '... (8772 chars total)",'
                        '"outputFiles":["/tmp/output/Pearsall_Play_Executive_Review.pptx"],'
                        '"inputFileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t8_4",
        "content": (
            '{"exitCode":0,"stdout":"Presentation saved: /tmp/output/'
            'Pearsall_Play_Executive_Review.pptx\\n","uploadedFiles":'
            '[{"fileName":"Pearsall_Play_Executive_Review.pptx","fileSize":261607}]}'
        ),
    },
    # Turn 8 response
    {
        "role": "assistant",
        "content": (
            "## Executive PPT Created\n\n"
            "I created an **executive-style PowerPoint** with benchmark comparison, "
            "top-well ranking, county/operator insights, and management implications.\n\n"
            "### Core message of the deck\n\n"
            "- **Pearsall is weaker on liquids** than the Eagle Ford West benchmark\n"
            "- **Pearsall is much gassier**, with higher GOR and lower liquids cut\n"
            "- The **best overall Pearsall well** is **WHITLEY-DUBOSE UNIT 1H**\n"
            "- **PALS RANCH 12H** is one of the strongest pure producers, but more gas-heavy\n"
            "- **Frio** and **Atascosa** are the first places to focus"
        ),
    },
    # == Turn 9: User asks for KI visuals ==
    {
        "role": "user",
        "content": (
            "can you give me ki visuals that helps me understand and comprehend "
            "the well characteristics specially for the pure cell play"
        ),
    },
    # == Turn 10: AI creates Pearsall-specific visual deck (4 tool calls) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Considering visual deliverables**\n\n"
            "I interpret 'pure cell play' as the pure Pearsall subset. I'll create "
            "a branded PPTX with various charts for well characteristics.\n```"
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
            '{"files":[{"fileRef":1,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.xlsx","fileSize":122563,"source":"user"},'
            '{"fileRef":2,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.csv.xlsx","fileSize":197488,"source":"user"},'
            '{"fileRef":6,"fileName":"Pearsall_Play_Executive_Review.pptx",'
            '"fileSize":261607,"source":"ai"}],"count":6}'
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
                        '{"code":"import pandas as pd, numpy as np, glob, os\\n'
                        "path = [p for p in glob.glob('/tmp/data/*') "
                        "if p.endswith('.xlsx')][0]\\n"
                        "df = pd.read_excel(path, sheet_name='BASE')\\n"
                        "df = df[(df['Trajectory']=='HORIZONTAL') & "
                        "(df['ENVPlay']=='PEARSALL')].copy()\\n"
                        "# Key medians, correlations, quartiles, top 12 wells\\n"
                        '... (2526 chars total)",'
                        '"language":"python","fileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t10_2",
        "content": (
            '{"exitCode":0,"stdout":"Pure Pearsall horizontal wells: 26\\n'
            "Counties: {'FRIO': 15, 'ATASCOSA': 6, 'LA SALLE': 4, 'MCMULLEN': 1}\\n"
            "Key medians:\\nCumOil_BBL 32346.0\\nCumGas_MCF 301178.0\\n"
            "CumProd_BOE 82542.0\\nFirst12MonthOil_BBL 18323.0\\n"
            "PeakOil_BBL 5799.0\\nLateralLength_FT 4781.5\\n"
            "GOR_ScfPerBbl 9692.0\\nWHLiquids_PCT 38.0\\n"
            "MonthsToPeakProduction 2.0\\n\\n"
            "Correlations to CumOil_BBL:\\n"
            "FlowingTubingPressure_PSI 0.44 n=22\\n"
            "WaterIntensity_GALPerFT 0.347 n=24\\n"
            "OilGravity_API 0.253 n=25\\n\\n"
            "CumOil_BBL quartiles {0.25: 22848.0, 0.5: 32346.0, 0.75: 42898.0}\\n"
            "GOR_ScfPerBbl quartiles {0.25: 4995.0, 0.5: 9692.0, 0.75: 16572.0}\\n"
            'WHLiquids_PCT quartiles {0.25: 27.0, 0.5: 38.0, 0.75: 55.0}"}'
        ),
    },
    # render_document for visual PPTX (charts + slides)
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
                        '{"format":"pptx","code":"import pandas as pd, numpy as np\\n'
                        "import matplotlib.pyplot as plt\\n"
                        "# Generate 6 charts: bubble plot, boxplots, scatter plots\\n"
                        "# Build branded PPTX with Pearsall well characteristics\\n"
                        '... (12000 chars total)",'
                        '"outputFiles":["/tmp/output/Pure_Pearsall_Key_Visuals.pptx"],'
                        '"inputFileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t10_3",
        "content": (
            '{"exitCode":0,"stdout":"Created Pure_Pearsall_Key_Visuals.pptx",'
            '"uploadedFiles":[{"fileName":"Pure_Pearsall_Key_Visuals.pptx",'
            '"fileSize":771206}]}'
        ),
    },
    # Turn 10 response
    {
        "role": "assistant",
        "content": (
            "## Pure Pearsall Key Visuals\n\n"
            "Created a branded visual deck for the **26 horizontal Pearsall wells**.\n\n"
            "Key charts include:\n"
            "- Cum oil vs GOR bubble plot\n"
            "- Oil/gas/BOE boxplots by county\n"
            "- Completion intensity scatter plots\n"
            "- Liquids cut distribution\n\n"
            "Key medians:\n"
            "- Cum oil: **32,346 bbl**\n"
            "- GOR: **9,692 scf/bbl**\n"
            "- Liquids cut: **38%**\n"
            "- Lateral length: **4,782 ft**"
        ),
    },
    # == Turn 11: User uploads Formentera MS Plot Observations PPTs ==
    {
        "role": "user",
        "content": (
            "Can you look at these two PPT files and give me an executive-style "
            "summary for this?"
        ),
    },
    # == Turn 12: AI reads PPTs and produces geological summary (2 tool calls) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Analyzing conversation files**\n\n"
            "I need to read the Formentera MS Plot Observations PPTs and produce "
            "an executive summary of the mass spec gas observations.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t12_1",
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
        "tool_call_id": "tc_t12_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.xlsx","fileSize":122563,"source":"user"},'
            '{"fileRef":2,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.csv.xlsx","fileSize":197488,"source":"user"},'
            '{"fileRef":8,"fileName":"Formentera MS Plot Observations - July 19 AM.pptx",'
            '"fileSize":573760,"source":"user"},'
            '{"fileRef":9,"fileName":"Formentera MS Plot Observations - July 21 AM.pptx",'
            '"fileSize":903140,"source":"user"}],"count":9}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t12_2",
                "type": "function",
                "function": {
                    "name": "read_conversation_files_batch",
                    "arguments": '{"fileRefs":[8,9],"maxCharsPerFile":120000}',
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t12_2",
        "content": (
            '{"files":[{"fileRef":8,"fileName":"Formentera MS Plot Observations - '
            'July 19 AM.pptx","content":"Formentera Darlene 1-STX N731HP\\n'
            "Mass Spec Observations: interval starts at 6000 ft. Methane dominates. "
            "Between 6500-6700 ft small gas peaks with C1-C4 increase. "
            "At top of Eagle Ford higher gas continues into Del Rio. Dry gas prominent "
            "with increase in heavier gases. Below Del Rio gas values decrease. "
            "From about 7900 ft slight decrease in Methane and rise in heavier gases. "
            "Helium jump below 7500 ft. Hydrogen increase at 7300 ft."
            '","charCount":2845},'
            '{"fileRef":9,"fileName":"Formentera MS Plot Observations - '
            'July 21 AM.pptx","content":"Formentera Darlene 1-STX N731HP\\n'
            "Updated observations: confirms July 19 trends. At 8200 ft gas equipment "
            "zeroed. Trip at 8961 ft generated high trip gas between 9000-9150 ft. "
            "H2 and He anomalies: 6500-7300 ft He and H2 covary. 7300-7500 ft H2 jumps. "
            "7500-8500 ft all three gases covary. 8500-8961 ft H2 jumps above He."
            '","charCount":4401}],"totalFiles":2}'
        ),
    },
    # Turn 12 response: Formentera executive summary
    {
        "role": "assistant",
        "content": (
            "## Summary\n\n"
            "Across the two Formentera mass spec observation decks, the main story is "
            "a **gas system that is mostly methane-dominant at shallower intervals**, "
            "strengthens materially through the **Eagle Ford-Del Rio section**, then "
            "shifts to relatively **heavier gas signatures deeper in the hole**.\n\n"
            "### Executive Takeaways\n\n"
            "1. **Strongest hydrocarbon indication: Eagle Ford-Del Rio interval**\n"
            "2. Shallower section is comparatively simple and methane-led\n"
            "3. Deeper section trends heavier from about **7,900 ft**\n"
            "4. **Helium and Hydrogen anomalies** are real but not yet definitive\n"
            "5. **Operational effects** are material (zeroing, blowbacks, "
            "MWD trip at **8,961 ft**)\n\n"
            "### What Changed from July 19 to July 21\n\n"
            "| Topic | July 19 | July 21 Update |\n"
            "|---|---|---|\n"
            "| Eagle Ford-Del Rio | High gas, dry gas dominant | Same, reaffirmed |\n"
            "| Helium anomaly | Possible fault/fracture | Reinterpreted as likely "
            "porosity/permeability |\n"
            "| Hydrogen anomaly | Possibly formation-related | Later H2 rises may be "
            "drilling/tool effects |\n"
            "| Late high gas | Uncertain due to maintenance | Tied to trip gas after "
            "MWD trip at 8,961 ft |"
        ),
    },
    # == Turn 13: User asks about Eagle Ford relationship ==
    {
        "role": "user",
        "content": (
            "Is this in any way related to the Eagle Ford play or wells that we "
            "analyzed before in this conversation? If so can you build up the "
            "relationship between these two files that we just observed and the "
            "Eagle Ford wells that we analyzed before?"
        ),
    },
    # == Turn 14: AI builds relationship analysis (2 tool calls) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Analyzing conversation files**\n\n"
            "I need to check if Darlene/Formentera/N731HP appears in the earlier "
            "spreadsheet. I'll search and compare the two data types.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t14_1",
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
        "tool_call_id": "tc_t14_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.xlsx","fileSize":122563},'
            '{"fileRef":2,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.csv.xlsx","fileSize":197488},'
            '{"fileRef":8,"fileName":"Formentera MS Plot Observations - July 19 AM.pptx",'
            '"fileSize":573760},'
            '{"fileRef":9,"fileName":"Formentera MS Plot Observations - July 21 AM.pptx",'
            '"fileSize":903140}],"count":9}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t14_2",
                "type": "function",
                "function": {
                    "name": "analyze_spreadsheet",
                    "arguments": (
                        '{"code":"import pandas as pd, glob, os\\n'
                        "path = [p for p in glob.glob('/tmp/data/*') "
                        "if p.endswith('.xlsx')][0]\\n"
                        "df = pd.read_excel(path, sheet_name='BASE')\\n"
                        "# Search for DARLENE, FORMENTERA, N731HP, EAGLE FORD, DEL RIO\\n"
                        "terms = ['DARLENE','FORMENTERA','N731HP','EAGLE FORD']\\n"
                        '... (1629 chars total)",'
                        '"language":"python","fileRefs":["2"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t14_2",
        "content": (
            '{"exitCode":0,"stdout":"ROWS: 82\\n'
            "TERM: DARLENE | HITS: 0\\n"
            "TERM: FORMENTERA | HITS: 0\\n"
            "TERM: N731HP | HITS: 0\\n"
            "TERM: EAGLE FORD | HITS: 50\\n\\n"
            "PLAY EAGLE FORD: n=55\\n"
            "Counties: {'LA SALLE': 23, 'MCMULLEN': 17, 'ATASCOSA': 10, 'FRIO': 5}\\n\\n"
            "PLAY PEARSALL: n=27\\n"
            "Counties: {'FRIO': 15, 'ATASCOSA': 6, 'LA SALLE': 5, 'MCMULLEN': 1}\"}"
        ),
    },
    # Turn 14 response: relationship analysis
    {
        "role": "assistant",
        "content": (
            "## Summary\n\n"
            "**Yes -- the two Formentera PPTs are related to the Eagle Ford play "
            "at the stratigraphic and petroleum-system level, but not directly tied "
            "to any specific well in the Eagle Ford/Pearsall spreadsheet set.**\n\n"
            "### Direct relationship to the play: Yes\n"
            "- PPTs describe gas behavior at the **top of the Eagle Ford** and "
            "through the **Del Rio**\n\n"
            "### Direct relationship to exact wells: No\n"
            "- DARLENE, FORMENTERA, N731HP not found in earlier spreadsheet\n\n"
            "### How the two analyses connect\n\n"
            "| Earlier analysis | Formentera PPTs |\n"
            "|---|---|\n"
            "| Production/completion dataset | Mud gas / mass spec interpretation |\n"
            "| Multi-well benchmark | Single-well subsurface observation |\n"
            "| What produced | What was encountered while drilling |\n\n"
            "### Key benchmark comparison\n"
            "- Eagle Ford benchmark: median GOR **1,528 scf/bbl**, liquids cut **80%**, "
            "cum oil **105,610 bbl**\n"
            "- Pearsall: median GOR **9,692 scf/bbl**, liquids cut **38%**, "
            "cum oil **32,346 bbl**\n"
            "- Formentera well: gas-forward, more consistent with Pearsall-type behavior\n\n"
            "### Bottom line\n"
            "The Formentera PPTs are **upstream, interval-scale evidence** from a well "
            "crossing the Eagle Ford system, while the earlier spreadsheet is "
            "**downstream, well-performance evidence** from Eagle Ford and Pearsall wells."
        ),
    },
    # == Turn 15: User requests one-page crosswalk ==
    {
        "role": "user",
        "content": (
            "Yes please do that. Build me a one-page crosswalk with three columns "
            "as you recommended"
        ),
    },
    # == Turn 16: AI creates crosswalk PDF (3 tool calls) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Analyzing and summarizing data**\n\n"
            "I'll create a one-page PDF crosswalk with three columns: "
            "Formentera MS observation, What it means geologically, "
            "How it ties to prior Eagle Ford/Pearsall analysis.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t16_1",
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
        "tool_call_id": "tc_t16_1",
        "content": (
            '{"files":[{"fileRef":1,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.xlsx","fileSize":122563},'
            '{"fileRef":2,"fileName":"Pearsall well vs 1st EF West Oil '
            'Metrics_2023-06-16.csv.xlsx","fileSize":197488},'
            '{"fileRef":8,"fileName":"Formentera MS Plot Observations - July 19 AM.pptx",'
            '"fileSize":573760},'
            '{"fileRef":9,"fileName":"Formentera MS Plot Observations - July 21 AM.pptx",'
            '"fileSize":903140}],"count":9}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t16_2",
                "type": "function",
                "function": {
                    "name": "read_conversation_files_batch",
                    "arguments": '{"fileRefs":[8,9],"maxCharsPerFile":80000}',
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t16_2",
        "content": (
            '{"files":[{"fileRef":8,"content":"Formentera Darlene 1-STX N731HP '
            "Mass Spec observations for July 19..."
            '","charCount":2845},'
            '{"fileRef":9,"content":"Formentera Darlene 1-STX N731HP '
            "Updated observations for July 21..."
            '","charCount":4401}],"totalFiles":2,"cacheHits":2}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t16_3",
                "type": "function",
                "function": {
                    "name": "render_document",
                    "arguments": (
                        '{"format":"pdf","code":"from reportlab.platypus import '
                        "Paragraph, Spacer, Table, TableStyle\\n"
                        "# Build 3-column crosswalk table: Formentera observation, "
                        "geological meaning, tie to prior analysis\\n"
                        '... (7083 chars total)",'
                        '"outputFiles":["/tmp/output/Formentera_to_EagleFord_Crosswalk.pdf"]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t16_3",
        "content": (
            '{"exitCode":0,"stdout":"Created /tmp/output/'
            'Formentera_to_EagleFord_Crosswalk.pdf","uploadedFiles":'
            '[{"fileName":"Formentera_to_EagleFord_Crosswalk.pdf","fileSize":13413}]}'
        ),
    },
    # Turn 16 response
    {
        "role": "assistant",
        "content": (
            "## One-Page Crosswalk Created\n\n"
            "Created a **3-column executive table** in PDF:\n\n"
            "1. **Formentera MS observation**\n"
            "2. **What it means geologically**\n"
            "3. **How it ties to prior Eagle Ford / Pearsall analysis**\n\n"
            "Core message: The Formentera PPTs are **clearly related to the Eagle Ford "
            "system** but are **not direct matches** to the exact wells in the prior "
            "benchmark spreadsheet. The right relationship is:\n\n"
            "Formentera = single-well subsurface fluid evidence\n"
            "Earlier Eagle Ford/Pearsall work = multi-well production outcome evidence"
        ),
    },
    # == Turn 17: User uploads additional files (H2S PDF, comparison images) ==
    {
        "role": "user",
        "content": (
            "can you also analyze these file and see how they correlate with the "
            "information we already have and create presentation executive style "
            "that gives the brief overview"
        ),
    },
    # == Turn 18: AI uses research_documents and asks clarification (3 tool calls) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Planning file analysis**\n\n"
            "I need to read the newly uploaded H2S PDF, Early Pearsall to Early EF PDF, "
            "and comparison images, then correlate with existing findings.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t18_1",
                "type": "function",
                "function": {
                    "name": "research_documents",
                    "arguments": (
                        '{"input":"Read and summarize the conversation files: '
                        "Formentera MS Plot Observations - July 19/21, "
                        "H2S Reading On PALS Ranch 12H 60ppm.pdf, and "
                        "Early Pearsall to Early EF.pdf. Focus on key observations "
                        "and connections to earlier spreadsheet findings."
                        '... (580 chars total)"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t18_1",
        "content": (
            '{"keyFindings":"1) Formentera decks describe gas-prone system: methane '
            "dominates, notable high-gas at Eagle Ford-Del Rio, heavier components "
            "below 7900 ft. 2) H2S certificate confirms PALS Ranch 12H in Indio Tanks "
            "(Pearsall), Frio County, BlackBrush O&G, 60 ppm H2S, max escape volume "
            "5,000 MCF/d. 3) Consistent with benchmark: Pearsall median GOR 9,692 vs "
            "EF 1,528 scf/bbl, liquids cut 38% vs 80%. 4) PALS Ranch 12H: 91,326 bbl "
            "cum oil, 53,880 bbl 12-mo oil, GOR 17,834 -- strong producer but very gassy. "
            '5) Early Pearsall to EF PDF extraction was poor quality.",'
            '"status":"partial"}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t18_2",
                "type": "function",
                "function": {
                    "name": "update_conversation_context",
                    "arguments": (
                        '{"context":"User comparing Pearsall vs Eagle Ford. Analyzed '
                        "spreadsheets, Formentera MS PPTs, H2S PDF. Current request: "
                        'create executive-style presentation."}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t18_2",
        "content": (
            '{"success":true,"message":"Conversation context saved."}'
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_t18_3",
                "type": "function",
                "function": {
                    "name": "ask_clarification",
                    "arguments": (
                        '{"questions":[{"question":"For the executive presentation, '
                        'what outcome do you want emphasized most?","options":['
                        '{"label":"Play comparison","value":"play_comparison"},'
                        '{"label":"PALS Ranch 12H case study","value":"pals_case_study"},'
                        '{"label":"Formentera geology","value":"formentera_geology"},'
                        '{"label":"Balanced overview","value":"balanced_overview"}]},'
                        '{"question":"How long should the deck be?","options":['
                        '{"label":"Very short (4-5 slides)","value":"short"},'
                        '{"label":"Standard executive (6-8 slides)","value":"standard"},'
                        '{"label":"Expanded (9-12 slides)","value":"expanded"}]}]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t18_3",
        "content": (
            '{"status":"waiting_for_user","questionCount":2}'
        ),
    },
    # Turn 18 response: clarification questions
    {
        "role": "assistant",
        "content": (
            "I can do that. Before I build the executive deck, I need two quick "
            "choices from you:\n\n"
            "### 1) What should the deck emphasize most?\n"
            "- Play comparison: Pearsall vs Eagle Ford\n"
            "- PALS Ranch 12H case study\n"
            "- Formentera geologic / fluid implications\n"
            "- Balanced overview of all files\n\n"
            "### 2) How long should the deck be?\n"
            "- Very short (4-5 slides)\n"
            "- Standard executive (6-8 slides)\n"
            "- Expanded (9-12 slides)"
        ),
    },
    # == Turn 19: User answers clarification ==
    {
        "role": "user",
        "content": (
            "Here are my answers to your clarification questions:\n\n"
            "For the executive presentation, what outcome do you want emphasized "
            "most?: formentera_geology\n"
            "How long should the deck be?: standard\n\n"
            "Please proceed with creating the plan based on these answers."
        ),
    },
    # == Turn 20: AI creates plan and deck (1 tool call -- create_plan) ==
    {
        "role": "assistant",
        "content": (
            "**Thought Process**:\n```\n**Planning analysis and presentation**\n\n"
            "I'm connecting observations to earlier spreadsheet metrics. "
            "Three-step process: (1) extract documents, (2) refresh benchmark, "
            "(3) generate PPT.\n```"
        ),
        "tool_calls": [
            {
                "id": "tc_t20_1",
                "type": "function",
                "function": {
                    "name": "create_plan",
                    "arguments": (
                        '{"reasoning":"Standard executive deck emphasizing Formentera '
                        "geology, tying to earlier Pearsall vs Eagle Ford benchmark "
                        'and PALS Ranch 12H case.",'
                        '"steps":[{"stepIndex":0,"title":"Extract and interpret new '
                        'uploaded documents","agentType":"document_search"},'
                        '{"stepIndex":1,"title":"Refresh benchmark linkage to prior '
                        'well analysis","agentType":"spreadsheet_analysis"},'
                        '{"stepIndex":2,"title":"Build executive presentation",'
                        '"agentType":"code_execution","dependsOn":[0,1]}]}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_t20_1",
        "content": (
            '{"status":"plan_created","stepCount":3}'
        ),
    },
    # Turn 20 response: plan and benchmark findings
    {
        "role": "assistant",
        "content": (
            "## Proposed Plan\n\n"
            "1. **Extract and interpret new uploaded documents**\n"
            "2. **Refresh benchmark linkage to prior well analysis**\n"
            "3. **Build executive presentation**\n\n"
            "## Key Benchmark Findings\n\n"
            "- Dataset: **82 wells** on BASE sheet: **27 Pearsall**, **55 Eagle Ford**, "
            "**140 columns**\n"
            "- **Eagle Ford is more oil-rich**, **Pearsall is more gas-weighted**\n\n"
            "### Median comparisons:\n"
            "- 12-month oil: EF **30.1 MBbl** vs Pearsall **18.3 MBbl** (EF = 1.64x)\n"
            "- 12-month gas: Pearsall **115.9 MMcf** vs EF **33.4 MMcf** (Pearsall = 3.47x)\n"
            "- 12-month BOE: Pearsall **42.6 MBOE** vs EF **35.8 MBOE**\n"
            "- 36-month oil: EF **57.1 MBbl** vs Pearsall **26.6 MBbl** (EF = 2.15x)\n"
            "- 36-month gas: Pearsall **237.3 MMcf** vs EF **73.5 MMcf** (Pearsall = 3.23x)\n\n"
            "### PALS Ranch 12H anchor well:\n"
            "- Frio County, Indio Tanks field, BlackBrush O&G\n"
            "- 12-mo: **53.9 MBbl oil**, **591.6 MMcf gas**, **152.5 MBOE**\n"
            "- 36-mo: **74.1 MBbl oil**, **1.118 Bcf gas**, **260.4 MBOE**\n"
            "- Outperformance vs Pearsall medians: **2.9x** oil, **5.1x** gas, "
            "**3.6x** BOE\n\n"
            "### Bottom line\n"
            "- **Eagle Ford = oil-rich analog**\n"
            "- **Pearsall = gas-rich analog**\n"
            "- **PALS Ranch 12H = best Pearsall anchor well**\n"
            "- Formentera geology presentation should frame around **which fluid family "
            "Formentera most resembles**\n\n"
            "Note: Document extraction (Step 1) failed due to specialist timeout. "
            "The planning deck structure is ready but the final built deck was not generated."
        ),
    },
]

# == Critical facts that MUST survive compression ==
# These are the numerical KPIs and identifiers that the agent's analysis depends on.
# If any are lost, the agent cannot produce correct summaries or presentations.
CRITICAL_FACTS: list[str] = [
    # Well identifiers
    "WHITLEY-DUBOSE UNIT 1H",
    "PALS RANCH 12H",
    "PALS RANCH 11H",
    "ZCW 1H",
    "ALINE WILLIAMS UNIT B 1H",
    "WELDON SCHORP-SANDSPUR UNIT 101H",
    "R.H. PICKENS",
    "WILSON C 1HP",
    "ALONZO PEELER A 1H",
    "ROBERT HINDES 1H",
    "CRAWFORD RANCH 12H",
    "Darlene 1-STX N731HP",
    # Play identifiers
    "Pearsall",
    "Eagle Ford",
    "Formentera",
    # County names
    "Frio",
    "Atascosa",
    "La Salle",
    "McMullen",
    # Operator names
    "ALVARADO 2014 LLC",
    "BLACKBRUSH O&G",
    "CHEYENNE PETROLEUM",
    "CRESCENT ENERGY",
    "EOG",
    # GOR values
    "9,692",       # Pearsall median GOR (scf/bbl)
    "1,528",       # Eagle Ford median GOR (scf/bbl)
    "3,557",       # WHITLEY-DUBOSE GOR
    "17,834",      # PALS RANCH 12H GOR
    # Cumulative oil values
    "95,467",      # WHITLEY-DUBOSE cum oil (bbl)
    "91,326",      # PALS RANCH 12H cum oil (bbl)
    "72,193",      # ZCW 1H cum oil (bbl)
    "32,346",      # Pearsall median cum oil (bbl)
    "105,610",     # Eagle Ford median cum oil (bbl)
    # 12-month oil values
    "48,201",      # WHITLEY-DUBOSE 12-mo oil
    "53,880",      # PALS RANCH 12H 12-mo oil
    "18,323",      # Pearsall median 12-mo oil
    "30,136",      # Eagle Ford median 12-mo oil
    # Viability scores
    "98.4",        # WHITLEY-DUBOSE viability score
    "83.7",        # PALS RANCH 12H viability score
    "82.6",        # ZCW 1H viability score
    # Liquids cut
    "38",          # Pearsall median liquids cut (%)
    "80",          # Eagle Ford median liquids cut (%)
    "63",          # WHITLEY-DUBOSE liquids cut (%)
    # Well counts
    "26",          # Pearsall well count
    "54",          # Eagle Ford well count (or 55 in some counts)
    "80",          # Total horizontal well count
    # Geological depth markers
    "7,900",       # Depth where heavier gas signature begins (ft)
    "8,961",       # MWD trip depth (ft)
    # H2S data
    "60",          # PALS Ranch 12H H2S (ppm)
    "5,000",       # H2S max escape volume (MCF/d)
    # Benchmark comparisons
    "1.64",        # Eagle Ford 12-mo oil advantage ratio
    "3.47",        # Pearsall 12-mo gas advantage ratio
    "2.15",        # Eagle Ford 36-mo oil advantage ratio
    # Production months to peak
    "2.0",         # Pearsall median months to peak
    "4.5",         # Eagle Ford median months to peak
    # Conclusion keywords
    "gas-rich",
    "oil-rich",
    "gassier",
]

# == LLM Judge prompt for quality evaluation ==
LLM_JUDGE_PROMPT = """\
You are evaluating the quality of a compressed AI conversation. The original
conversation was a 20-turn data analysis session ($6.91 cost) where an AI agent
analyzed two Pearsall oil play spreadsheets, benchmarked Pearsall vs Eagle Ford
wells, identified the most viable wells, created executive presentations, then
analyzed Formentera mass spec observations and built a crosswalk connecting the
geological observations to the production benchmark.

ORIGINAL CONVERSATION SUMMARY:
- Agent analyzed 80 horizontal wells (26 Pearsall, 54 Eagle Ford) from two spreadsheets
- Pearsall vs Eagle Ford key differences: median GOR 9,692 vs 1,528 scf/bbl,
  median cum oil 32,346 vs 105,610 bbl, median liquids cut 38% vs 80%
- Top Pearsall wells identified: WHITLEY-DUBOSE UNIT 1H (viability 98.4),
  PALS RANCH 12H (viability 83.7), ZCW 1H (viability 82.6)
- DOCX report, executive PPTX, and Pearsall key visuals created
- Formentera Darlene 1-STX N731HP mass spec PPTs analyzed: Eagle Ford-Del Rio
  high gas interval, heavier gas below 7,900 ft, H2/He anomalies
- Crosswalk built connecting Formentera geology to Eagle Ford/Pearsall benchmark
- H2S certificate for PALS Ranch 12H: 60 ppm, 5,000 MCF/d max escape
- Final plan for Formentera geology executive deck (partially completed)

COMPRESSED CONVERSATION:
{compressed_text}

ANCHOR LEDGER (preserved facts):
{anchor_ledger}

Evaluate on these criteria (score 1-5 each):

1. **Fact Retention**: Are the key numerical KPIs preserved? (well counts, GOR values,
   cum oil, viability scores, depth markers, H2S data)
2. **Decision Preservation**: Are the conclusions preserved? (Pearsall is gassier,
   WHITLEY-DUBOSE is most viable, Formentera is Eagle Ford system-related but not
   same wells)
3. **Tool Call Coherence**: Can you tell what tools were called and what they produced?
   Are tool call/result pairs intact? Are the failure-retry patterns visible?
4. **Task Flow**: Can you follow the conversation flow? (analyze spreadsheets ->
   DOCX report -> viability ranking -> executive PPT -> Formentera PPTs ->
   relationship analysis -> crosswalk -> H2S correlation -> final deck plan)
5. **Actionability**: Could an agent continue this conversation from the compressed
   version without re-reading the spreadsheets or PPTs?

Return JSON:
{{"fact_retention": <1-5>, "decision_preservation": <1-5>, "tool_coherence": <1-5>,
  "task_flow": <1-5>, "actionability": <1-5>, "overall": <1-5>,
  "missing_facts": ["list of important facts that were lost"],
  "assessment": "1-2 sentence summary"}}
"""
