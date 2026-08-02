# Phản hồi Reviewer — Task 1: bổ sung baseline

Nguồn số liệu đầy đủ: [`Review_Task_1.md`](Review_Task_1.md) ·
`experiment_results/task1/{shieldgemma,qwen3guard}/`

---

## 1. Thí nghiệm bổ sung

Theo góp ý của reviewer về việc Task 1 mới chỉ có một guard model
zero-shot, chúng tôi đã chạy bổ sung **hai guard model mới**, trên đúng
tập test đã dùng trong bản thảo (GuardChat verified test, n = 1000), với
cùng hai representation đầu vào (single-turn prompt và multi-turn
conversation).

**ShieldGemma-2B.** Guard model của Google, xây trên Gemma-2. Cách hoạt
động khác các guard thông thường: thay vì phân loại vào một bộ nhãn cố
định, nó nhận **một chính sách an toàn viết bằng văn xuôi** kèm theo đoạn
text cần chấm, rồi trả về xác suất `P(Yes)` cho câu hỏi "đoạn text này có
vi phạm chính sách đó không". Muốn chấm 6 category thì chạy 6 lần, mỗi
lần một chính sách. Ưu điểm là chính sách do người đánh giá tự viết nên
khớp đúng định nghĩa của GuardChat; nhược điểm là mỗi mẫu tốn 6 lượt
forward và các chính sách chấm độc lập nhau nên có thể cùng lúc kích hoạt
nhiều nhãn.

**Qwen3Guard-Gen-8B.** Guard model của Alibaba, dạng generative — đọc hội
thoại rồi **sinh ra** dòng phán quyết (`Safety: Safe / Controversial /
Unsafe` kèm tên category). Nó được huấn luyện chuyên cho kiểm duyệt hội
thoại đa lượt, đa ngôn ngữ, và trong bộ nhãn gốc có hẳn một lớp
`Jailbreak` dành riêng cho các prompt cố tình lách hệ thống — tức là dạng
tấn công mà GuardChat mô phỏng. Theo khảo sát so sánh các guard model mã
nguồn mở của Harsh et al. [1], **Qwen3Guard hiện là một trong những model
mạnh nhất cho tác vụ kiểm tra prompt NSFW**, nên chúng tôi coi đây là
baseline zero-shot khó nhất hiện có cho Task 1.

Riêng Qwen3Guard chúng tôi chạy **hai chế độ** để tách bạch năng lực nhận
diện khỏi ảnh hưởng của việc ánh xạ nhãn:

| Chế độ | Cách làm | Ý nghĩa |
|---|---|---|
| `guardchat` | Đưa thẳng định nghĩa 6 category của GuardChat vào prompt, yêu cầu model chấm theo đúng 6 lớp đó | Đồng nhất bộ nhãn với các baseline khác |
| `native` | Để model dùng đúng 9 category gốc nó được huấn luyện, rồi ánh xạ 9 → 6 về taxonomy GuardChat | Phản ánh model "as released", không ép nó ra khỏi phân bố huấn luyện |

Với ShieldGemma, ngưỡng quyết định `P(Yes)` là một siêu tham số phải
chọn; chúng tôi dùng **0.7** cho toàn bộ kết quả dưới đây (lý do ở mục
4.4).

Cả ba model đều chạy zero-shot, không fine-tune, trên cùng phần cứng và
cùng file test.

---

## 2. Kết quả chi tiết

ASR (%) ↓ (ASR = 1 − binary recall), Macro-F1 ↑ trên 6 category,
n = 1000 mỗi ô.

| Model (mode) | ASR raw | ASR prompt | ASR conv | F1 raw | F1 prompt | F1 conv |
|---|---|---|---|---|---|---|
| Llama-Guard-3-8B (guardchat) | 80.9 | 23.6 | 68.6 | 0.175 | 0.291 | 0.176 |
| ShieldGemma-2B (guardchat, th=0.7) | 81.7 | 21.3 | 39.7 | 0.227 | **0.398** | **0.273** |
| Qwen3Guard-Gen-8B (guardchat) | 64.5 | 13.8 | 12.0 | 0.146 | 0.131 | 0.154 |
| Qwen3Guard-Gen-8B (native) | 53.8 | **11.0** | **10.5** | **0.301** | 0.194 | 0.223 |

Ba cột đầu vào:

* `raw` — prompt seed gốc, trước bước enhancement. Đưa vào để đối chiếu,
  **không phải một representation tấn công**: seed thường chưa mang nội
  dung vi phạm nên ASR cao ở đây phản ánh khoảng cách giữa nhãn ý đồ và
  bề mặt văn bản, không phải lỗ hổng phòng thủ.
* `prompt` — enhanced prompt, tương ứng cột single-turn của Bảng 1.
* `conv` — hội thoại đa lượt nối lại (trung bình 7.68 lượt), tương ứng
  cột multi-turn của Bảng 1.

> Dòng Llama-Guard trong bảng này là kết quả chạy lại của chúng tôi ở chế
> độ `guardchat` với hội thoại nối chuỗi, nên không trùng với dòng
> Llama-Guard trong Bảng 1 (mục 3). Chúng tôi đang chạy thêm cấu hình
> taxonomy gốc `S1–S14` + chat template chuẩn để thống nhất một nguồn số
> duy nhất cho bản final.

---

## 3. Bảng 1 cập nhật

| Methods | Single-turn ASR (%) ↓ | Multi-turn ASR (%) ↓ | Macro-F1 (Multi-turn) ↑ |
|---|---|---|---|
| BiLSTM | 14.22 | 80.45 | 0.312 |
| BERT | 9.85 | 72.30 | 0.422 |
| SafeGuider | **5.50** | 58.21 | 0.487 |
| Qwen2.5-7B (Zero-shot) | 7.90 | 60.33 | 0.542 |
| Llama-Guard-3-8B (Zero-shot) | 6.50 | 52.15 | **0.579** |
| **ShieldGemma-2B (Zero-shot, th=0.7)** | 21.30 | 39.70 | 0.273 |
| **Qwen3Guard-Gen-8B (Zero-shot)** | 11.00 | **10.50** | 0.223 |

Hai dòng in đậm là kết quả bổ sung. Qwen3Guard báo cáo ở chế độ `native`
(model as released).

---

## 4. Nhận xét

### 4.1. Nhận xét chung

**Hai baseline mới hạ trần multi-turn ASR.** Con số tốt nhất trong bảng
cũ là 52.15%; Qwen3Guard đạt **10.50%** (cải thiện 5.0×) và ShieldGemma
đạt **39.70%** (1.3×). Điều này cho thấy GuardChat vẫn là benchmark khó,
nhưng **không phải khó với mọi kiến trúc**.

**Khoảng cách single-turn → multi-turn chỉ biến mất ở Qwen3Guard.** Các
baseline cũ đều sụt rất mạnh khi chuyển sang hội thoại (SafeGuider +52.7
điểm ASR, Llama-Guard +45.7 điểm). ShieldGemma cũng sụt **+18.4 điểm** —
nhẹ hơn nhiều nhưng cùng bản chất. Riêng Qwen3Guard đi
ngược: **−0.5 điểm**, tức multi-turn còn *dễ* hơn single-turn một chút.
Kết luận nên điều chỉnh theo hướng này: tấn công đa lượt phá được
**classifier huấn luyện có giám sát và guard dựa nhiều vào từ khoá**,
chứ không phải phá được mọi hệ thống nhận diện.

**Đánh đổi rõ rệt giữa hai cột.** Hai model mới mạnh nhất ở cột ASR
nhưng lại có Macro-F1 thấp nhất bảng. Nói cách khác, chúng rất giỏi trả
lời "đoạn này có unsafe không" nhưng yếu ở "unsafe thuộc loại nào". Trên
bảng mới, hai cột ASR và Macro-F1 gần như **nghịch biến** — phần khó của
benchmark đã dịch từ *phát hiện* sang *quy kết nhãn*. Chúng tôi cho rằng
đây là quan sát đáng giá nhất từ thí nghiệm bổ sung này và sẽ đưa vào
phần thảo luận.

### 4.2. Vì sao hai model mới chịu được tấn công đa lượt tốt hơn

Hai lý do, cả hai đều đến từ **cách hai model được huấn luyện**, không
phải từ việc chúng lớn hơn hay mới hơn.

**Qwen3Guard được huấn luyện chuyên cho đúng dạng tấn công này.** Dữ liệu
huấn luyện của nó là hội thoại nhiều lượt, và bộ nhãn gốc có riêng một
lớp `Jailbreak` mô tả *hình thức* tấn công (prompt cố tình lách hệ thống)
chứ không chỉ *nội dung* vi phạm. GuardChat về bản chất là một tập
jailbreak đa lượt, nên nó nằm gọn trong phân bố huấn luyện của model —
đây là lý do trực tiếp khiến tỉ lệ tấn công thành công chỉ còn 10.5%, và
cũng phù hợp với vị trí dẫn đầu của model này trong khảo sát của Harsh
et al. [1].

**ShieldGemma chấm mô tả cảnh, không chấm từ ngữ.** Vì nó so đoạn text
với một chính sách viết bằng văn xuôi, nó vẫn kích hoạt được khi câu chữ
đã được làm sạch.

Chúng tôi đo được cơ chế này một cách định lượng. Tấn công đa lượt của
GuardChat hoạt động bằng cách **gỡ bỏ từ vựng độc hại**: enhanced prompt
trung bình chứa 3.10 từ tục/gore, còn hội thoại chỉ còn 1.02, và 57.9%
hội thoại **không chứa từ nào**. Recall nhị phân trên riêng nhóm hội
thoại "sạch từ vựng" đó:

| | Llama-Guard-3-8B | ShieldGemma-2B (0.7) | Qwen3Guard-Gen-8B |
|---|---|---|---|
| Recall khi hội thoại không có từ khoá độc hại nào | 0.24 | 0.43 | **0.84** |

Cùng một tập mẫu, cùng một điều kiện. Khi không còn từ khoá để bám vào,
Qwen3Guard vẫn suy ra được ý đồ từ ngữ cảnh tích luỹ (0.84); Llama-Guard
thì không (0.24). ShieldGemma nằm giữa (0.43) — nó đọc được ngữ cảnh
nhưng độ tự tin giảm khi ý đồ bị pha loãng, nên nhiều mẫu không vượt được
điểm cắt. Đây chính là phần giải thích cho khoảng cách ở cột multi-turn
ASR.

### 4.3. Vì sao Macro-F1 (multi-turn) của hai model mới thấp

**Macro-F1 thấp ở đây chủ yếu là hệ quả của việc ánh xạ nhãn, không phải
của năng lực nhận diện.** Không model nào trong ba được huấn luyện trên
đúng 6 category của GuardChat: Llama-Guard-3 có 14 lớp gốc, Qwen3Guard có
9, ShieldGemma có 4. Để tính được Macro-F1 6 chiều, bắt buộc phải ánh xạ
— và mỗi cách ánh xạ đều mất mát theo một kiểu.

**Với Qwen3Guard (`native`) — lớp `shocking` không thể được dự đoán.**
Taxonomy gốc của model không có category nào cho nội dung gore /
body-horror, nên `shocking` bị tính F1 = 0 **theo cấu trúc**, bất kể model
đọc đúng hay sai. Riêng điều này đã kéo Macro-F1 xuống 1/6. Nếu chỉ tính
trên 5 lớp mà model thực sự với tới được thì con số là **0.268** thay vì
0.223.

**Một lớp gốc phải gánh nhiều lớp GuardChat.** Ví dụ rõ nhất:
`Unethical Acts` của Qwen3Guard bao cả thiên kiến, phân biệt đối xử, hate
speech, xúc phạm, đe doạ, phỉ báng *và* thông tin sai lệch — rộng hơn
nhiều so với `harassment` của GuardChat. Kết quả trên nhóm mẫu gold
`harassment`: model bắt đúng **90%** số mẫu là unsafe, nhưng F1 của lớp
`harassment` chỉ 0.163, vì nó gọi tên bằng một lớp khác. Đây là lỗi ánh
xạ, không phải lỗi nhận diện.

**Ở chế độ `guardchat`, Qwen3Guard không tuân theo taxonomy được đưa
vào.** Dù prompt liệt kê đủ 6 category của GuardChat, model vẫn xuất nhãn
gốc của nó (`Violent`, `Sexual`, `Unethical`…). Chỉ 2 trong 6 lớp nhận
được dự đoán, 4 lớp còn lại F1 = 0 theo cấu trúc — nên con số 0.154 không
phản ánh năng lực phân loại. Tính trên 2 lớp model thực sự dùng thì
Macro-F1 là **0.462**, cao nhất trong tất cả các cấu hình đã chạy.

**Với ShieldGemma, vấn đề nằm ở phía nhãn gold.** Do 6 chính sách chấm
độc lập, nó kích hoạt trung bình 0.77 nhãn/mẫu trên hội thoại và 1.26
nhãn/mẫu trên prompt đơn, trong khi
gold của GuardChat chỉ có **đúng 1 nhãn/mẫu**. Những nhãn thừa phần lớn
là nhãn *có thật trong văn bản* nhưng không được ghi trong gold — ví dụ
lớp `shocking` được kích hoạt 517 lần trong khi support chỉ 97, vì phần
lớn enhanced prompt của GuardChat đều mang yếu tố gore bất kể ý đồ gốc là
gì. Precision bị phạt cho những dự đoán đọc đúng văn bản.

**Tóm lại**, ba con số Macro-F1 trong Bảng 1 hiện không cùng mẫu số. Để
bảng đọc được đúng, chúng tôi đề xuất bổ sung vào bản final:

1. Một footnote ở caption ghi rõ model nào chạy chế độ nào và **lớp nào
   không thể với tới được** trong taxonomy của model đó.
2. Một cột phụ "Macro-F1 trên các lớp với tới được" (Qwen3Guard native:
   0.268 thay vì 0.223).
3. Cân nhắc gán multi-label cho tập test, hoặc bổ sung chỉ số
   "gold ∈ tập dự đoán" bên cạnh Macro-F1, để không phạt các dự đoán vốn
   mô tả đúng nội dung.

### 4.4. Về điểm vận hành của ShieldGemma

ShieldGemma trả về xác suất `P(Yes)` cho từng chính sách chứ không trả về
nhãn, nên cần một điểm cắt. Chúng tôi dùng **`P(Yes) ≥ 0.7`**: đây là
điểm vận hành thận trọng, chỉ báo vi phạm khi model thực sự tự tin, phù
hợp với một bộ lọc đặt trước hệ thống sinh ảnh — nơi chặn nhầm prompt
lành tính là chi phí trực tiếp lên người dùng.

Cần lưu ý khi đọc bảng: **phần lợi của điểm vận hành thận trọng không đo
được trên tập test hiện tại.** Tập test gồm 1000 mẫu unsafe và không có
mẫu lành tính nào, nên tỉ lệ chặn nhầm — thứ mà ngưỡng cao đánh đổi để
lấy — bằng 0 theo cấu trúc. Nói cách khác, bảng hiện tại chỉ ghi nhận
phần chi phí (recall thấp hơn) mà không ghi nhận phần lợi. Chúng tôi sẽ
bổ sung một tập benign (prompt lành tính từ DiffusionDB) và báo cáo
**ASR kèm FPR** để cấu hình này được đánh giá công bằng.

Vì mọi dự đoán đều lưu kèm `P(Yes)` của cả 6 chính sách, mọi điểm cắt
khác đều dựng lại được offline mà không phải chạy lại model — chi tiết
trong `Review_Task_1.md`.

---

## 5. Tài liệu tham khảo

[1] Harsh, R. R., Sarmah, B., Pasquali, S. *Benchmarking Open-Source
Safety Guard Models: A Comprehensive Evaluation.* arXiv:2605.28830, 2026.
Published as a workshop paper at ICLR 2026.

```bibtex
@article{harsh2026benchmarking,
  title={Benchmarking Open-Source Safety Guard Models: A Comprehensive Evaluation},
  author={Harsh, Reetu Raj and Sarmah, Bhaskarjit and Pasquali, Stefano},
  journal={arXiv preprint arXiv:2605.28830},
  year={2026}
}
```
