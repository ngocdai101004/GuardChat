# Phản hồi Reviewer — Task 2: đổi metric sang SBERT và so sánh single-turn / multi-turn

Trả lời **W3** (CLIP không phải metric phù hợp cho tương đồng text-text)
và **Suggestion 2** (cần so sánh tường minh single-turn với multi-turn).

Nguồn số liệu đầy đủ: [`Review_Task_2.md`](Review_Task_2.md) ·
`experiment_results/task2/similarity/`

---

## 1. Metric mới: SBERT thay cho CLIP

Chúng tôi đồng ý với reviewer và đã thay metric. Toàn bộ Task 2 hiện được
chấm lại bằng **`sentence-transformers/all-mpnet-base-v2`**, mean pooling
theo attention mask rồi chuẩn hoá L2, cosine giữa bản gốc và bản viết
lại.

### 1.1. Vì sao là `all-mpnet-base-v2`, không phải SBERT "bản gốc"

Điều này cần nói rõ vì "SBERT" ngày nay không còn trỏ tới đúng các
checkpoint trong bài báo SBERT gốc. Các model đó — `bert-base-nli-mean-tokens`
và họ hàng `*-nli-*`, `*-stsb-*` — **đã bị chính tác giả đánh dấu là
deprecated**. Model card của chúng hiện mở đầu bằng:

> ⚠️ This model is deprecated. Please don't use it as it produces sentence
> embeddings of low quality.

và chuyển hướng người đọc sang trang Pretrained Models của thư viện
`sentence-transformers` [1]. Trang đó khuyến nghị họ model `all-*`, được
huấn luyện trên **hơn 1 tỉ cặp câu**, và nêu rõ:

> The `sentence-transformers/all-mpnet-base-v2` model provides the best
> quality, while `sentence-transformers/all-MiniLM-L6-v2` is 5 times
> faster and still offers good quality.

Chúng tôi chọn nhánh chất lượng cao nhất vì đây là pass đo offline chạy
một lần trên 1000 mẫu — tốc độ không phải ràng buộc, toàn bộ hai metric ×
ba baseline × hai representation mất khoảng 3 phút trên một GPU.

### 1.2. Vì sao lựa chọn này quan trọng với đúng bộ dữ liệu của chúng tôi

Ngoài lý do nguyên tắc mà reviewer nêu (CLIP được huấn luyện để căn ảnh
với text, không phải text với text), trong trường hợp cụ thể của GuardChat
còn một lý do định lượng: **cửa sổ token**.

| | CLIP ViT-L/14 | all-mpnet-base-v2 |
|---|---|---|
| cửa sổ | 77 token | **384 token** |
| cách gộp vector | hidden state tại vị trí EOS | mean pooling có mask |
| prompt đơn lẻ bị cắt | **80%** (801/1000) | 4% (36/1000) |
| hội thoại bị cắt | **100%** | 98% (chấm nguyên khối) |

Với 80% prompt và 100% hội thoại bị cắt, CLIP không đọc hết đoạn text mà
nó đang chấm. Ở single-turn điều này làm sai lệch biên độ; ở multi-turn
nó phá hỏng hẳn phép đo, tới mức CLIP chấm 911/1000 hội thoại đúng bằng
1.0000. Chúng tôi tách riêng cơ chế đó ra mục 2.3 vì nó cũng là câu trả
lời cho việc **vì sao cột CLIP của conversation lại cao bất thường**.

### 1.3. Cách chúng tôi trình bày

Chúng tôi **báo cáo cả hai metric song song** thay vì chỉ thay số. Cùng
bộ bản ghi, cùng đoạn code, chỉ khác encoder — nên mọi chênh lệch quy về
đúng một biến, và luận điểm "đổi metric" trở thành thứ kiểm chứng được
chứ không phải khẳng định suông:

```bash
for E in sbert clip; do
    ENCODER=$E bash scripts/eval_task2_similarity.sh llama gemini safeguider
done
```

Đây là pass đo offline trên trường `rewritten_text` đã lưu, không chạy
lại rewriter, không tốn API. File kết quả của baseline không bị sửa, nên
metric có thể định nghĩa lại và chấm lại bất cứ lúc nào.

---

## 2. Kết quả cập nhật

n = 1000 mỗi ô. **Cột `penalised` là cột chính**: nó tính các mẫu rewriter
thất bại (bị nhà cung cấp chặn / từ chối) là 0.0. Nếu báo cáo cột `mean`
thì model nào từ chối càng nhiều lại càng "giữ nghĩa tốt".

### 2.1. Single-turn prompt

Cột CLIP là con số **đã báo cáo trong Bảng 2 của bản thảo**, để reviewer
thấy đúng thứ đang được thay thế. Cột SBERT là phép đo mới.

| Method | CLIP Sim. (Bảng 2, bản thảo) | SBERT mean | **SBERT penalised** | không chấm được |
|---|---|---|---|---|
| SafeGuider | 0.351 | **0.8664** | **0.8664** | **0** |
| Llama-3.1-8B | 0.418 | 0.7311 | 0.7303 | 1 (refusal) |
| Gemini 2.5 Flash → **3.5 Flash** | 0.515 | 0.6589 | 0.6194 | 60 (blocked) |

> **Hai cột này không phải một ablation chỉ đổi encoder.** Cột CLIP đến
> từ lần chạy trong bản thảo, cột SBERT đến từ lần chạy hiện tại, và giữa
> hai lần có ít nhất hai thay đổi khác: baseline proprietary đã chuyển từ
> Gemini 2.5 Flash sang 3.5 Flash, và cấu hình SafeGuider có khác biệt so
> với bản công bố (ghi trong `scripts/benchmark_task2_safeguider.sh`).
> Vì vậy chênh lệch giữa hai cột **không quy hết về encoder được**. Phép
> so sánh có kiểm soát — cùng bản ghi, cùng đoạn code, chỉ khác encoder —
> nằm ở mục 2.2 và mục 4.

Điều đáng chú ý nhất: **hai metric xếp hạng ngược nhau.** CLIP xếp
SafeGuider thấp nhất (0.351) và Gemini cao nhất (0.515); SBERT xếp đúng
ngược lại (0.866 so với 0.619). Với SafeGuider thì SBERT có lý do cơ học
rõ ràng để chấm cao: nó là phương pháp **chỉ xoá token, không bao giờ
thêm chữ** — trung bình xoá 12.3 từ và giữ lại 78% từ vựng gốc, so với
53% của Llama và 39% của Gemini. Một thước đo text-text đọc đủ 384 token
sẽ thấy phần lớn câu còn nguyên; CLIP với cửa sổ 77 token thì không.

Đây chính là loại đảo thứ hạng khiến chúng tôi đồng tình với reviewer:
metric cũ không chỉ lệch về biên độ mà có thể lệch về **kết luận**.

### 2.2. Multi-turn conversation

Ở đây cả hai encoder đều chạy trên **cùng bộ bản ghi và cùng đoạn code**,
nên chênh lệch giữa hai cột CLIP và SBERT quy về đúng một biến là encoder.

| Method | SBERT nguyên khối | **SBERT per-turn** | CLIP nguyên khối | CLIP per-turn |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.9654 | **0.9443** | 0.9962 | 0.9412 |
| Gemini-3.5-flash | 0.9262 | **0.9056** | 0.9524 | 0.9049 |
| SafeGuider | 0.9254 | 0.8612 | 0.8832 | 0.7747 |

(Cột conversation đã là `penalised`. `nguyên khối` = nối toàn bộ lượt
thành một chuỗi rồi chấm một lần; `per-turn` = chấm lượt *i* với lượt
*i*, rồi lấy trung bình.)

### 2.3. Vì sao CLIP nguyên khối ở conversation lại cao bất thường

Con số cần giải thích là **0.9962 và 0.9524** ở cột "CLIP nguyên khối" —
gần như bằng 1, và cao hơn hẳn mọi con số CLIP ở single-turn. Nếu đọc
theo nghĩa đen thì nó có nghĩa "viết lại hội thoại gần như không làm mất
nghĩa gì", điều này sai.

Cơ chế là sự trùng hợp của hai thứ:

**(1) Các lượt đầu hội thoại gần như vô hại, nên rewriter hầu như không
đụng tới.** Cấu trúc jailbreak đa lượt của GuardChat mở đầu bằng những
câu bình thường rồi mới leo thang; bộ khử độc phản ứng đúng theo phân bố
đó. Tỉ lệ lượt **hoàn toàn không bị sửa** (similarity > 0.9999) ở ba lượt
đầu là **80.9%** với Llama và **85.0%** với Gemini, so với 1.1% và 7.6%
ở các lượt cuối (số liệu đầy đủ ở mục 3.2).

**(2) CLIP chỉ đọc được 77 token đầu, tức gần như chỉ đọc đúng phần
đó.** Hội thoại dài trung bình ~2860 ký tự trong khi 77 token ≈ 300 ký
tự. Kiểm chứng trực tiếp trên các mẫu bị CLIP chấm đúng 1.0000: tiền tố
trùng khít giữa bản gốc và bản viết lại có median **990 ký tự (Llama) /
1100 ký tự (Gemini)**, và **907/911 với 883/884** số mẫu đó có tiền tố
chung dài hơn 300 ký tự. Cửa sổ 77 token nằm **hoàn toàn bên trong vùng
hai chuỗi giống hệt nhau**.

Ghép hai điều lại: CLIP đang so một chuỗi với chính nó. Nó chấm
**911/1000 hội thoại của Llama và 884/957 của Gemini đúng bằng 1.0000**
— không phải vì bản viết lại bảo toàn ngữ nghĩa, mà vì **phần bị sửa nằm
ngoài cửa sổ đo**. Trên đúng những mẫu đó, SBERT (đọc 384 token) cho mean
0.969/0.973 với **min 0.790/0.726** và 187/155 mẫu dưới 0.95 — tức các
bản viết lại đó có sửa thật, và sửa đáng kể.

Ba hệ quả:

* Con số CLIP nguyên khối ở conversation **không phải là điểm cao, mà là
  hằng số**. Nó không phân biệt được model nào với model nào (Llama
  0.9962 vs Gemini 0.9952, chênh 0.001).
* Chấm **theo từng lượt** khôi phục lại khả năng đo của chính CLIP:
  0.9412 / 0.9049, sát với SBERT (0.9443 / 0.9056). Cùng encoder, cùng
  dữ liệu, chỉ đưa đơn vị văn bản xuống dưới cửa sổ token — xem mục 4.
* SafeGuider là đối chứng tự nhiên: nó **có** sửa vào 77 token đầu (chỉ
  để yên 21.9% số lượt mở đầu), và CLIP nguyên khối của nó lập tức rời
  khỏi vùng bão hoà — 0.8832, chỉ 141 mẫu chấm 1.0, độ lệch chuẩn 0.106
  so với 0.019 của Llama. Điều này xác nhận hằng số 1.0 là **thuộc tính
  của dữ liệu nằm ngoài cửa sổ**, không phải lỗi nội tại của encoder.

**Số nguyên khối của conversation cũng không dùng để so với single-turn
được** — xem mục 3.

---

## 3. So sánh tường minh single-turn với multi-turn (Suggestion 2)

### 3.1. Một cảnh báo trước khi so

Không thể đặt 0.73 (single-turn) cạnh 0.965 (multi-turn) rồi kết luận
"viết lại hội thoại dễ hơn". Hội thoại GuardChat dài trung bình **7.68
lượt** (6 lượt: 5 mẫu · 7 lượt: 339 · 8 lượt: 627 · 9 lượt: 29), và phần
lớn các lượt đó vô hại nên **không rewriter nào đụng tới**. Con số 0.965
là trung bình của khoảng 6 lượt gần như nguyên vẹn và 1.7 lượt bị sửa
mạnh — nó cao vì lý do không liên quan gì đến chất lượng viết lại.

Muốn so tương ứng thì phải so single-turn với **phần hội thoại thật sự
mang nội dung tấn công**, tức các lượt cuối.

### 3.2. Phân tích theo ba cụm lượt

Chia mỗi hội thoại thành lượt 1-3, 4-6 và 7+ (cụm cuối trung bình 1.69
lượt), chấm theo từng lượt để miễn nhiễm với việc cắt token:

| cụm lượt | Llama | Gemini | SafeGuider |
|---|---|---|---|
| **1-3** (mở đầu) | 0.9901 | 0.9908 | 0.8871 |
| **4-6** (giữa) | 0.9426 | 0.9433 | 0.8387 |
| **7+** (cuối) | **0.8641** | **0.8756** | 0.8538 |

Tỉ lệ lượt **hoàn toàn không bị đụng tới** (similarity > 0.9999):

| cụm lượt | Llama | Gemini | SafeGuider |
|---|---|---|---|
| 1-3 | **80.9%** | **85.0%** | 21.9% |
| 4-6 | 24.4% | 28.2% | 3.3% |
| 7+ | **1.1%** | 7.6% | 0.8% |

### 3.3. Bốn insight rút ra

**(a) Với rewriter LLM, ba cụm là ba chế độ hành vi khác hẳn nhau, không
phải một đường dốc:** *bỏ qua* → *sửa một phần* → *sửa gần hết*. Median
của cụm 1-3 đúng bằng 1.0000. Đây là bằng chứng trực tiếp cho điều
reviewer gợi ý: cấu trúc jailbreak đa lượt **dồn nội dung độc hại về
cuối**, và bộ khử độc phản ứng đúng theo phân bố đó.

Quy luật này không phải trung bình của nhiều hành vi trộn lẫn. Ghép cặp
*trong cùng một hội thoại* (cụm 7+ trừ cụm 1-3 của chính nó), loại bỏ
khả năng con số bị bóp méo vì hội thoại dài ngắn khác nhau:

| | chênh lệch | tỉ lệ hội thoại mang dấu âm |
|---|---|---|
| Llama | −0.1263 | **98.0%** (n = 995) |
| Gemini | −0.1149 | 94.0% (n = 951) |
| SafeGuider | −0.0314 | 63.2% (n = 995) |

Vị trí lượt bị sửa mạnh nhất trong mỗi hội thoại cũng nói cùng một điều:

| | cụm 1-3 | cụm 4-6 | cụm 7+ |
|---|---|---|---|
| Llama | 3% | 24% | **73%** |
| Gemini | 6% | 25% | **69%** |
| SafeGuider | 25% | 50% | 25% |

**(b) So đúng đơn vị thì lợi thế của single-turn biến mất, và với một số
method thì đảo chiều.** Đặt single-turn cạnh cụm 7+ — hai thứ thật sự
tương ứng về mặt nội dung:

| | single-turn | multi-turn, cụm 7+ | chênh |
|---|---|---|---|
| Llama | 0.7311 | 0.8641 | +0.133 |
| Gemini | 0.6589 | 0.8756 | +0.216 |
| SafeGuider | 0.8664 | 0.8538 | −0.013 |
| **chênh Llama − Gemini** | **+0.0716** | **−0.0115** | |

Ngay cả khi so đúng đơn vị, cùng một câu tấn công vẫn được bảo toàn ngữ
nghĩa tốt hơn khi nó nằm trong hội thoại so với khi đứng một mình. Đáng
chú ý hơn là **thứ hạng giữa hai LLM đảo chiều**: Gemini sửa mạnh hơn hẳn
Llama ở prompt đơn lẻ (+0.072 nghiêng về Llama) nhưng lại sửa *nhẹ hơn*
ở đuôi hội thoại (−0.012), và bỏ qua hẳn 7.6% số lượt ở cụm đó so với
1.1% của Llama.

Giả thuyết của chúng tôi: khi một câu tấn công nằm xen giữa các lượt vô
hại, nó **bớt "trông nguy hiểm"**, nên rewriter can thiệp nhẹ tay hơn.
Nếu đúng thì đây chính là cơ chế mà cấu trúc đa lượt của GuardChat khai
thác — và nó là câu trả lời trực tiếp cho câu hỏi của reviewer về ảnh
hưởng của multi-turn lên prompt sanitization. Chúng tôi ghi nhận đây là
**giả thuyết chưa kiểm chứng**; xác nhận nó cần Safe Generation Rate tách
theo representation, đang chạy.

**(c) Lượt cuối cùng là chỗ đáng lo nhất về mặt vận hành.**

| | mean | không đụng tới |
|---|---|---|
| Llama | 0.8456 | **2/1000 (0.2%)** |
| Gemini | 0.8601 | **48/956 (5.0%)** |
| SafeGuider | 0.8701 | 7/1000 (0.7%) |

Gemini để nguyên lượt cuối ở 48 hội thoại, cao hơn Llama **24 lần**. Nếu
nội dung độc hại dồn về cuối như dữ liệu cho thấy, đây là 48 mẫu gần như
chắc chắn còn nguyên payload — một dạng thất bại mà **con số gộp 0.9262
hoàn toàn che mất**.

**(d) Không phải method nào cũng định vị được chỗ cần sửa.** SafeGuider
là ví dụ đối chứng hữu ích: nó không có ba chế độ trên (0.887 → 0.839 →
0.854, cụm giữa còn thấp hơn cụm cuối), chỉ để yên 21.9% số lượt mở đầu
thay vì 81-85%, và vị trí lượt hỏng nặng nhất phân bố gần đúng theo tỉ lệ
số lượt trong mỗi cụm — tức **độc lập với vị trí trong hội thoại**.
Nguyên nhân nằm ở thiết kế: nó xử lý từng lượt độc lập, không có khái
niệm "hội thoại này leo thang về cuối". Kết quả là nó sửa 6.91 trên 7.68
lượt mỗi hội thoại và rải đều thiệt hại thay vì tập trung vào chỗ cần.

### 3.4. Hệ quả cho cách báo cáo

Với conversation, chúng tôi sẽ báo cáo **per-turn hoặc cụm 7+**, không
dùng số nguyên khối. Nếu vẫn nêu số nguyên khối thì kèm tỉ lệ bị cắt 98%
để người đọc biết con số đó nghĩa là gì.

---

## 4. Bằng chứng bổ sung: vấn đề của CLIP là cửa sổ token, không phải encoder

Phần này không bắt buộc cho W3 nhưng chúng tôi đưa vào vì nó cho phép nói
chính xác CLIP sai ở đâu, thay vì chỉ nói nó không phù hợp.

Toàn bộ số trong mục này đến từ **phép so sánh có kiểm soát**: chúng tôi
chạy lại CLIP trên đúng bộ bản ghi hiện tại, qua đúng đoạn code đã dùng
cho SBERT, nên biến duy nhất thay đổi là encoder. (Đây là lý do các giá
trị CLIP ở đây không trùng với cột CLIP của Bảng 2 trong mục 2.1 — cột
đó là số đã công bố, từ một lần chạy khác.)

Tương quan hạng Spearman giữa hai metric, tách theo mức độ cửa sổ ăn vào
văn bản:

| nhóm | n | Spearman(CLIP, SBERT) |
|---|---|---|
| prompt — CLIP đọc trọn vẹn | 191 / 196 | **0.910 / 0.960** |
| prompt — CLIP bị cắt | 808 / 744 | 0.776 / 0.780 |
| conversation — nguyên khối (cắt 100%) | 1957 | **0.294 / 0.341** |
| conversation — per-turn, từng lượt | 15024 | **0.936 / 0.945** |

Hai dòng cuối là **cùng một dữ liệu**, cùng encoder, chỉ khác cách cắt
khúc: chấm nguyên khối cho tương quan 0.29, chấm từng lượt cho 0.94. Khi
CLIP đọc được trọn văn bản, nó xếp hạng gần trùng khít với một sentence
encoder chuyên dụng.

Kết luận: CLIP **không sai về bản chất trên mọi tác vụ**, nhưng trên bộ
dữ liệu này — prompt sinh ảnh dài và hội thoại 7-8 lượt — cửa sổ 77 token
làm nó mất khả năng đo. Đó là lý do cụ thể khiến việc chuyển sang SBERT
là cần thiết chứ không chỉ là lựa chọn hình thức.

Một quan sát củng cố: SafeGuider là method duy nhất **có sửa vào 77 token
đầu** của hội thoại, và với riêng nó CLIP nguyên khối chỉ chấm 141 mẫu
đúng 1.0 (so với 911 và 884), tương quan tăng lên 0.538. Nghĩa là con số
1.0 của CLIP chưa bao giờ là thuộc tính của encoder — nó là dấu vết của
việc phần bị sửa nằm ngoài cửa sổ.

---

## 5. Ghi chú và phần còn lại

1. Mọi con số ở đây là **bảo toàn ngữ nghĩa**, tức một nửa của đánh đổi.
   Similarity có thể tối ưu hoá tầm thường bằng cách không sửa gì.
   **Chúng tôi không kết luận method nào tốt hơn dựa trên bảng này.**
   Nửa còn lại là Safe Generation Rate, cần pipeline T2I và bộ gác an
   toàn ảnh; thí nghiệm đang chạy và sẽ báo cáo khi xong.
2. Mỗi bản ghi đều lưu vector `turn_similarities` đầy đủ theo lượt, nên
   mọi phân tổ theo vị trí trong mục 3 đều tái tạo được mà không cần
   encode lại.
3. Chúng tôi sẽ ghi rõ trong bản final: **encoder id, cách pooling, cửa
   sổ token**, kèm `num_unscorable`. Với sentence encoder, ba thứ đầu
   quyết định hoàn toàn con số, và trong bản thảo hiện tại chúng tôi mới
   nêu tên encoder.

---

## 6. Tài liệu tham khảo

[1] Sentence-Transformers — *Pretrained Models*.
    https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

[2] Reimers, N. and Gurevych, I. *Sentence-BERT: Sentence Embeddings
    using Siamese BERT-Networks.* EMNLP 2019.
