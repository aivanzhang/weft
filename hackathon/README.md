# Hackathon reference hub

Reference archive and analysis for the Meta Global AI Developer Hackathon. Collected October 26, 2026, using the user's event tab and official public pages. **No submissions, credit claims, agreement acceptances, posts, or messages were made.**

## Start here

| Need | File |
| --- | --- |
| What the judges want; fit of our application | [Fit and strategy](analysis/fit-and-strategy.md) |
| Requirements, Los Angeles schedule, unresolved conflicts | [Requirements](analysis/requirements.md) |
| Bounded demo, model roles, independent evaluation | [Demo plan](analysis/demo-plan.md) |
| What predates the event versus what is new | [Originality ledger](analysis/originality.md) |
| User application context | [Application context](analysis/application-context.md) |
| Every captured/unavailable reference | [Source index](SOURCE_INDEX.md) |
| First-hop link coverage and gaps | [Link inventory](LINKS.md) |
| Machine-readable URLs, dates, hashes, and status | [Source manifest](source-manifest.json) |

## Primary source documents

- [Event description](sources/event.md)
- [Participant guide: full substantive sections and tables](sources/participant-guide.md)
- [Official rules and waiver](sources/official-rules.md)
- [Actual submission form fields](sources/submission-form.md)
- [Event-page announcements](sources/announcements.md)
- [Meta credit-terms PDF converted to Markdown, all three pages](sources/credit-terms.md)
- [Discord access instructions](sources/discord-browser.md)
- [Livestream page and recording link](sources/livestream-browser.md)

The source index also covers all directly linked model pages, API docs, cookbook landing/directory pages, Meta video pages, partner homepages/docs, unfilled credit forms, host/speaker links, and linked legal material. References and analysis are different: snapshots record what the sources said; analysis records our interpretation and proposed decisions.

## Scope and known limits

- This is a **first-hop reference archive**, not a recursive mirror of every website linked by every documentation page. The API terms, acceptable-use policy, and credit PDF were additionally included because the credit form/rules directly incorporate them.
- Text, headings, code examples, tables where convertible, and links are preserved. Website chrome may remain in some documentation captures. Dynamic widgets, images, audiovisual content, and live behavior are not reproduced by Markdown.
- The participant guide and submission interface required rendered-browser inspection; their files state the manual capture method. Raw unauthenticated pages sometimes contained only loading shells, which are explicitly marked limited and supplemented by browser snapshots.
- Meta video landing-page text is captured, **not the spoken recordings/transcripts**. The kickoff player/recording URL is recorded, but no full transcript is available in this archive.
- One speaker LinkedIn page returned HTTP 999. Other LinkedIn pages expose partial public content and sign-in prompts; they are marked limited, not complete profiles.
- The direct Meta privacy link did not expose substantive text through the HTTP capture. The credit-terms link in the form redirects to a PDF and is archived using PDF text extraction. The PDF linked by the official waiver was also downloaded and converted independently.
- Discord membership UI was inspected; private channel history was not archived. No role was claimed or account reconnected.
- Original source timezone labels are retained, including inconsistent October “EST” labels. Use the [Los Angeles schedule](analysis/requirements.md) for planning.
- Generic site navigation, sign-in links, and user profile navigation are not followed as event resources. The link inventory records substantive outgoing links from primary event material.
- Copyright and source ownership remain with the source authors. Keep this reference collection private unless redistribution rights are reviewed. Public hackathon code and private research material can be separated later.

## Refresh procedure

Tools required: Python 3 standard library and `pandoc`. PDF extraction uses `pdftotext`. No package installation or browser-session token export is needed.

```sh
# Download only newly added targets; all network requests are GET.
python3 scripts/archive_hackathon.py --missing

# Re-convert existing raw captures without network requests.
python3 scripts/archive_hackathon.py --offline

# Explicitly refresh public HTTP captures; previous Markdown is backed up locally.
python3 scripts/archive_hackathon.py --refresh

# Rebuild the index and direct-link coverage report.
python3 scripts/index_hackathon.py
```

Browser snapshots, application transcription, and PDF conversion are not silently refreshed by the HTTP script. Revisit their exact source pages read-only when needed, preserve their prior version, and record a new capture date. Do not claim those files were refreshed merely because the public-page script ran.

Before using a deadline or restriction for an irreversible action, re-check the relevant official source. Record conflicts rather than editing original snapshots to make them agree. No scheduled refresh automation was created.
