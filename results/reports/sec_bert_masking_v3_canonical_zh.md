# SEC-BERT masking v3（canonical checkpoint 對齊）

日期：2026-10-22

## 實驗定位

本實驗只回答一個敏感性問題：模型表現有多少依賴原文中的顯式詞彙線索。它不是新的主模型，也不改變文章的 selective automation 主線。

- 主模型權重在遮罩測試前已固定；原文與遮罩文本使用相同的 890 個 retrospective test 樣本。
- 遮罩模型獨立使用 `text_context_masked_v2` 訓練，僅按 validation macro-F1 選擇 epoch。
- calibration/test 標籤不參與權重或超參數選擇。
- 差值置信區間使用 report-clustered paired bootstrap（2,000 次）。

## 結果

| 模型與輸入 | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Canonical model，原文 | 0.9337 | 0.9021 | 0.9058 |
| Canonical model，遮罩文本 | 0.3809 | 0.2787 | 0.3117 |
| 遮罩文本訓練模型，遮罩文本 | 0.6708 | 0.5332 | 0.5463 |

固定主模型在遮罩後的 prediction change rate 為 0.6191。Accuracy 差值（遮罩減原文）為 -0.5528，report-clustered 95% bootstrap interval 為 [-0.6034, -0.5036]；macro-F1 差值為 -0.6234，95% interval 為 [-0.7004, -0.5895]。

## 投稿解讀

結果支持「顯式詞彙線索對分類非常重要」的敏感性結論，但不能解讀為因果識別，也不能證明模型完全依賴單一關鍵詞。遮罩後重新訓練只恢復部分效能，適合作為 robustness/sensitivity evidence，放在主結果之後或補充材料中，不應成為文章主線。

## 可追溯產物

- Canonical checkpoint ZIP：`sec_bert_canonical_v3_checkpoint.zip`（作為版本化 release asset 發布時使用）
  - SHA256：`46AF65999D260CC5A100AE4F6DE99B6AF8E4798F67CE75160074E24F54396045`
- Masked checkpoint ZIP：`sec_bert_masked_v3_checkpoint.zip`（作為版本化 release asset 發布時使用）
  - SHA256：`2A74CA96DCF023B7B671681988C24C3319C50C27F531D469274052D46CD3108C`
- Canonical model weights SHA256：`BF265638D28B84745A7C89FADCA33E6ADBA5E280574ECA05938837B9CEF91128`
- Masked model weights SHA256：`552C2F72D4AE7696F2E601D9FB20F8C691D7008645AE15AB4566F1EF1A9B4F0B`
- 成對評估：本地 run 目錄不隨程式庫發布；其表格與指標位於版本化結果產物中。
- 遮罩模型完整 run：本地 run 目錄不隨程式庫發布；其 manifest 與摘要位於版本化結果產物中。

