# Multipart source inventory

Use the **textured original robot plus standalone dragon** as the primary local assembler inputs. They total **81,091 vertices, 87,821 triangles, 48,447,108 source bytes** and **75,497,472 texture pixels** across six images. They fit 100k vertices/triangles, 64 MiB source sum, 16 Mi-pixels per image and the newly specified 96 Mi-pixel aggregate cap. No pixel decoding or native memory-fit claim was made.

Exact original paths, full hashes, source counts, PBR/UV0 routes and rights statements are in `inventory-summary.json`; config/plan snapshots are retained. Both inputs are standalone unrigged generated characters with no clips. Metre/+Y-up storage conventions do not prove source physical scale. This standalone dragon is not the panda’s original shoulder companion.

The original untextured stages are optional geometry-only references (3,123,952 bytes together), not substitutes for textured evidence. Their equality with the textured stages was **not established**.

## Preserved harness failures

V1 incorrectly required an explicit identity matrix on optional untextured GLBs. V2 correctly admitted the omitted identity default; all four per-source checks completed, then an additional raw-vs-textured hash-equality assertion failed. The script did not retain the differing subject/attribute. Both failures and exited-process receipts remain unchanged. No third inspection run, source change, weakened geometry bound or false passing aggregate result was used.

This folder is a source inventory, not an assembled avatar or quality/animation acceptance. Future native work still needs its own 2 GiB bound and measured result.
