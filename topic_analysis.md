# Topic Analysis Across Platforms

This report compares bertopic results across different platforms for each ct keyword set. For each topic, I present top 10 topic (if any) distribution viz. topics are represented by top 10 words from the topic using tf-idf scores, min cluster size is set to 20, embedding model is `mnpnet-base-v2`, and used UMAP + HDBSCAN with default parameters. noise is handled automatically by HDBSCAN.

**Dataset Statistics:**
- Total Records: 5,490,661
- Missing Values: 4,741,220 (86.35%)

**Platform Distribution:**

| Platform     | Number of Posts |
|--------------|----------------|
| X (Twitter)  | 279,630        |
| Truth Social | 186,506        |
| Gab          | 112,722        |
| 4chan        | 62,638         |
| Bluesky      | 53,242         |
| GETTR        | 42,091         |
| Fediverse    | 12,612         |

*Note: Data has been filtered to remove missing values and non-English content before topic modeling.*

## Table of Contents
- [9/11 Conspiracy Topics](#911-conspiracy-topics)
- [Alien/UFO Conspiracy Topics](#alienufo-conspiracy-topics)
- [Moon Landing Conspiracy Topics](#moon-landing-conspiracy-topics)
- [COVID-19 Conspiracy Topics](#covid-19-conspiracy-topics)
- [New World Order Topics](#new-world-order-topics)

## 9/11 Conspiracy Topics

<table>
<tr>
<td width="33%">4chan<br><img src="topic_viz_KEYWORDS_9_11_4chan.png" alt="4chan 9/11 Topics"></td>
<td width="33%">X (Twitter)<br><img src="topic_viz_KEYWORDS_9_11_X.png" alt="X 9/11 Topics"></td>
<td width="33%">Bluesky<br><img src="topic_viz_KEYWORDS_9_11_bluesky.png" alt="Bluesky 9/11 Topics"></td>
</tr>
<tr>
<td width="33%">Gab<br><img src="topic_viz_KEYWORDS_9_11_gab.png" alt="Gab 9/11 Topics"></td>
<td width="33%">GETTR<br><img src="topic_viz_KEYWORDS_9_11_gettr.png" alt="GETTR 9/11 Topics"></td>
<td width="33%">Truth Social<br><img src="topic_viz_KEYWORDS_9_11_truthsocial.png" alt="Truth Social 9/11 Topics"></td>
</tr>
</table>

[View detailed topic comparison](KEYWORDS_9_11_platform_comparison.csv)

## Alien/UFO Conspiracy Topics

<table>
<tr>
<td width="33%">4chan<br><img src="topic_viz_KEYWORDS_ALIEN_4chan.png" alt="4chan Alien Topics"></td>
<td width="33%">X (Twitter)<br><img src="topic_viz_KEYWORDS_ALIEN_X.png" alt="X Alien Topics"></td>
<td width="33%">Bluesky<br><img src="topic_viz_KEYWORDS_ALIEN_bluesky.png" alt="Bluesky Alien Topics"></td>
</tr>
<tr>
<td width="33%">Fediverse<br><img src="topic_viz_KEYWORDS_ALIEN_fediverse.png" alt="Fediverse Alien Topics"></td>
<td width="33%">GETTR<br><img src="topic_viz_KEYWORDS_ALIEN_gettr.png" alt="GETTR Alien Topics"></td>
<td width="33%">Truth Social<br><img src="topic_viz_KEYWORDS_ALIEN_truthsocial.png" alt="Truth Social Alien Topics"></td>
</tr>
</table>

[View detailed topic comparison](KEYWORDS_ALIEN_platform_comparison.csv)

## Moon Landing Conspiracy Topics

<table>
<tr>
<td width="33%">4chan<br><img src="topic_viz_KEYWORDS_MOON_4chan.png" alt="4chan Moon Landing Topics"></td>
<td width="33%">X (Twitter)<br><img src="topic_viz_KEYWORDS_MOON_X.png" alt="X Moon Landing Topics"></td>
<td width="33%">Bluesky<br><img src="topic_viz_KEYWORDS_MOON_bluesky.png" alt="Bluesky Moon Landing Topics"></td>
</tr>
<tr>
<td width="33%">Gab<br><img src="topic_viz_KEYWORDS_MOON_gab.png" alt="Gab Moon Landing Topics"></td>
<td width="33%">GETTR<br><img src="topic_viz_KEYWORDS_MOON_gettr.png" alt="GETTR Moon Landing Topics"></td>
<td width="33%">Truth Social<br><img src="topic_viz_KEYWORDS_MOON_truthsocial.png" alt="Truth Social Moon Landing Topics"></td>
</tr>
</table>

[View detailed topic comparison](KEYWORDS_MOON_platform_comparison.csv)

## COVID-19 Conspiracy Topics

<table>
<tr>
<td width="33%">4chan<br><img src="topic_viz_KEYWORDS_COVID19_4chan.png" alt="4chan COVID-19 Topics"></td>
<td width="33%">X (Twitter)<br><img src="topic_viz_KEYWORDS_COVID19_X.png" alt="X COVID-19 Topics"></td>
<td width="33%">Bluesky<br><img src="topic_viz_KEYWORDS_COVID19_bluesky.png" alt="Bluesky COVID-19 Topics"></td>
</tr>
<tr>
<td width="33%">Fediverse<br><img src="topic_viz_KEYWORDS_COVID19_fediverse.png" alt="Fediverse COVID-19 Topics"></td>
<td width="33%">Gab<br><img src="topic_viz_KEYWORDS_COVID19_gab.png" alt="Gab COVID-19 Topics"></td>
<td width="33%">GETTR<br><img src="topic_viz_KEYWORDS_COVID19_gettr.png" alt="GETTR COVID-19 Topics"></td>
</tr>
<tr>
<td width="33%">Truth Social<br><img src="topic_viz_KEYWORDS_COVID19_truthsocial.png" alt="Truth Social COVID-19 Topics"></td>
<td width="33%"></td>
<td width="33%"></td>
</tr>
</table>

[View detailed topic comparison](KEYWORDS_COVID19_platform_comparison.csv)

## New World Order Topics

<table>
<tr>
<td width="33%">4chan<br><img src="topic_viz_KEYWORDS_NWO_4chan.png" alt="4chan NWO Topics"></td>
<td width="33%">X (Twitter)<br><img src="topic_viz_KEYWORDS_NWO_X.png" alt="X NWO Topics"></td>
<td width="33%">Bluesky<br><img src="topic_viz_KEYWORDS_NWO_bluesky.png" alt="Bluesky NWO Topics"></td>
</tr>
<tr>
<td width="33%">Fediverse<br><img src="topic_viz_KEYWORDS_NWO_fediverse.png" alt="Fediverse NWO Topics"></td>
<td width="33%">Gab<br><img src="topic_viz_KEYWORDS_NWO_gab.png" alt="Gab NWO Topics"></td>
<td width="33%">GETTR<br><img src="topic_viz_KEYWORDS_NWO_gettr.png" alt="GETTR NWO Topics"></td>
</tr>
<tr>
<td width="33%">Truth Social<br><img src="topic_viz_KEYWORDS_NWO_truthsocial.png" alt="Truth Social NWO Topics"></td>
<td width="33%"></td>
<td width="33%"></td>
</tr>
</table>

[View detailed topic comparison](KEYWORDS_NWO_platform_comparison.csv)

---

## Notes on Interpretation

- Each visualization shows the distribution of topics within that platform for the given conspiracy theory keyword set
- Topics are represented by different colors and labeled with their most representative words
- Closer points in the visualization indicate more similar topics
- The size of each cluster indicates the number of documents in that topic
- For detailed topic distributions and word lists, refer to the CSV files linked above each visualization set