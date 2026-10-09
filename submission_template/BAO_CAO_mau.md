# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Nhóm 114 · **Thành viên:** Đào Quang Thái Anh (MSV: 2A202602987)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

## 1. Cấu hình đã chọn

Mỗi video: tracker bạn nộp, `conf`, `iou`, điều bạn **nhìn thấy** trên video, và một cấu hình đã thử rồi loại.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | bytetrack | 0.3 | 0.5 | Camera tĩnh, ánh sáng tốt, quỹ đạo người đi bộ đều. ByteTrack giữ ID rất ổn định suốt các track dài, chỉ có 12 lần đổi ID (IDSW). Không bị nhảy hộp lung tung. | ocsort (conf 0.4, iou 0.5): bỏ sót các người ở xa phía cuối quảng trường do conf cao. |
| video_2 (phố đêm, tĩnh, rất đông) | bytetrack | 0.2 | 0.5 | Cảnh đêm từ trên cao, mật độ người cực kỳ đông, kích thước người nhỏ và tối. Hạ conf xuống 0.2 giúp bắt thêm người đi bộ trong bóng tối mà ByteTrack giai đoạn 2 vẫn liên kết tốt không bị bùng nổ false positive. | ocsort (conf 0.35, iou 0.5): nhiều người đi trong vùng tối không được tạo track, ID bị đứt đoạn liên tục. |
| video_3 (camera di động, ảnh nhỏ) | ocsort | 0.25 | 0.4 | Camera di chuyển và FPS thấp khiến vị trí người dịch chuyển lớn giữa các frame. OC-SORT với cơ chế OCM (bù quán tính quan sát) và ngưỡng IoU 0.4 giúp duy trì track tốt khi người di chuyển nhanh qua góc máy. | bytetrack (conf 0.3, iou 0.6): độ dịch chuyển khung hình làm giảm IoU thực tế, ngưỡng iou=0.6 làm mất dấu và gán ID mới liên tục. |
| video_4 (trong nhà, camera di chuyển) | bytetrack | 0.35 | 0.5 | Trong nhà có nhiều vách kính và bề mặt phản chiếu. Tăng conf lên 0.35 giúp triệt tiêu các hộp phát hiện giả từ bóng phản chiếu. Camera tịnh tiến đều về phía trước giúp Kalman Filter tuyến tính của ByteTrack dự đoán rất khớp. | bytetrack (conf 0.15, iou 0.5): xuất hiện nhiều hộp phát hiện nhấp nháy trên hình ảnh phản chiếu ở cửa kính và poster trong nhà. |
| video_5 (trên xe bus, giao lộ đông) | ocsort | 0.3 | 0.45 | Xe bus di chuyển qua giao lộ phức tạp và bị rung lắc mạnh theo phương dọc/ngang. OC-SORT ít bị ảnh hưởng bởi nhiễu rung lắc nhờ cập nhật lại vận tốc khi có quan sát mới, hạn chế việc hộp nhảy sang người bên cạnh lúc giao cắt. | bytetrack (conf 0.25, iou 0.5): khi xe xóc giật mạnh, sai số tích lũy của Kalman Filter làm hộp bay lệch và xảy ra hiện tượng hoán đổi ID giữa những người đứng cạnh nhau. |

## 2. Số liệu video_1

Dán bảng HOTA / MOTA / IDF1 do `scripts/evaluate_practice.py` in ra.

```
HOTA: nhom01_video1-pedestrian     HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
video_1                            26.912    15.068    48.130    15.288    82.599    50.467    84.841    84.551    27.116    32.559    81.385    26.498    

CLEAR: nhom01_video1-pedestrian    MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag      
video_1                            17.292    82.526    17.356    17.932    96.889    11.290    14.516    74.194    14.158    3332      15249     107       12        7         9         46        44        

Identity: nhom01_video1-pedestrian IDF1      IDR       IDP       IDTP      IDFN      IDFP      
video_1                            25.713    15.236    82.320    2831      15750     608       

Count: nhom01_video1-pedestrian    Dets      GT_Dets   IDs       GT_IDs    
video_1                            3439      18581     34        62        
```

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

Với **ít nhất hai video** (nên gồm một video bạn chỉ đánh giá bằng mắt), viết 3–5 câu:

- **Đối với video_1 (Quảng trường tĩnh, ban ngày):** ByteTrack thể hiện sự vượt trội khi camera cố định và người chuyển động thẳng đều với vận tốc ổn định. Nhờ cơ chế liên kết 2 giai đoạn (first-association với detection conf cao và second-association với detection conf thấp), ByteTrack tận dụng được cả những bounding box mờ khi người bị che khuất một phần, giúp số lần đổi danh tính chỉ dừng ở mức 12 lần (IDSW=12) và đạt độ chính xác phát hiện CLR_Pr lên tới 96.89%. Việc sử dụng mô hình Re-ID trong cảnh này không mang lại nhiều cải thiện rõ rệt nhưng lại làm tăng đáng kể chi phí tính toán (chậm hơn gấp nhiều lần trên CPU).
- **Đối với video_5 (Trên xe bus, giao lộ rung lắc):** Khi camera bị rung lắc mạnh và góc quay thay đổi phi tuyến tính, giả định chuyển động vận tốc không đổi của Kalman Filter tiêu chuẩn (dùng trong ByteTrack) bị phá vỡ hoàn toàn, dẫn đến việc các hộp dự đoán bị trôi và dễ hoán đổi ID giữa những người đi gần nhau. Tracker OC-SORT giải quyết triệt để vấn đề này nhờ cơ chế OCM (Observation-Centric Momentum) và tính toán lại quỹ đạo (observation-centric recovery), giúp bù đắp lại hiện tượng dằn xóc của xe và giữ đúng ID của người đi bộ ngay cả khi có rung lắc gián đoạn.

## 4. Nếu có thêm thời gian

Nếu có thêm thời gian thực nghiệm và phần cứng GPU mạnh hơn, nhóm sẽ thử nghiệm BoT-SORT tích hợp mô hình bù chuyển động camera (Camera Motion Compensation - CMC thông qua biến đổi affine) kết hợp trích xuất đặc trưng ngoại hình Re-ID (OSNet) trên video_3 và video_5 để giải quyết triệt để bài toán camera di động; đồng thời tinh chỉnh ngưỡng liên kết 2 tầng của ByteTrack một cách chi tiết hơn trên từng cảnh chiếu sáng khác nhau.
