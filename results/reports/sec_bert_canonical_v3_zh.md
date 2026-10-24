# SEC-BERT canonical v3：稿件數據凍結記錄

日期：2026-10-24

本記錄是正文、摘要、表格及圖注中 SEC-BERT 數字的最新版單一來源。模型底座為 `nlpaueb/sec-bert-base`，解析版本為 `d511591f5e74052afdab08f1f14c4ff2a1e55749`。訓練使用原始配方：最大長度 256、三個 epochs、learning rate 2e-5、physical batch size 4、gradient accumulation 4、class-weighted cross-entropy、seed 20261002；gradient checkpointing 只用於降低顯存，最佳 epoch 僅由 clean validation macro-F1 選擇。

## 基礎分類結果

Canonical checkpoint 在 890 筆 clean retrospective test 上取得 accuracy 0.9337079、macro-F1 0.9021008、balanced accuracy 0.9057970。相對 TF--IDF/context baseline，accuracy 高 0.0213483；5,000 次 report-cluster bootstrap 95% interval 為 [-0.0088604, 0.0525887]。Macro-F1 高 0.0485404，對應 interval 為 [0.0058200, 0.1495579]。模型保存後重新載入的最大 logits 絕對差為 0。

## 校準與凍結 policy

Temperature 僅在 608 筆 calibration findings 上擬合，凍結值為 1.15177542。Calibration NLL 由 0.3206223 降至 0.3112757，multiclass Brier score 由 0.1272625 降至 0.1244839，10-bin ECE 由 0.0459080 降至 0.0311030；accuracy 及 macro-F1 分別維持 0.9292763 和 0.9173709。

High-coverage threshold 為 0.5269052934。在 test 上轉介 32/890 筆，referral rate 0.0359551、coverage 0.9640449、selective accuracy 0.9522145、error capture 18/59（0.3050847）、referral precision 0.5625。Coverage 的 report-cluster 95% interval 為 [0.9521344, 0.9749746]，selective accuracy interval 為 [0.9337584, 0.9713116]。

Conservative threshold 為 0.8157442331。在 test 上轉介 63/890 筆，referral rate 0.0707865、coverage 0.9292135、selective accuracy 0.9661427、error capture 31/59（0.5254237）、referral precision 0.4920635。Coverage interval 為 [0.9108979, 0.9476525]，selective accuracy interval 為 [0.9461220, 0.9825996]。

## Checkpoint 與運行環境

Canonical model weights SHA256 為 `BF265638D28B84745A7C89FADCA33E6ADBA5E280574ECA05938837B9CEF91128`。完整 archive 名稱為 `sec_bert_canonical_v3_checkpoint.zip`，大小 406,116,411 bytes，SHA256 為 `46AF65999D260CC5A100AE4F6DE99B6AF8E4798F67CE75160074E24F54396045`。Archive 包含模型、tokenizer、訓練設定、環境、validation/calibration/test 預測、溫度、兩個凍結閾值、重現腳本及逐檔雜湊清單；其是否作為 release asset 發布取決於模型授權與檔案大小限制。

正式運行環境為 Windows 11、Python 3.13.1、PyTorch 2.11.0+cu128、Transformers 5.8.0、scikit-learn 1.9.0，以及 NVIDIA GeForce RTX 3050 Laptop GPU。

Masking v3 的結果與 archive 另見 `results/reports/sec_bert_masking_v3_canonical_zh.md`。
