
pain points: too many class in BigDL and SSD

    case class ByteRecord(data: Array[Byte], label: Float)
    class LabeledSentence[T: ClassTag](protected var _data: Array[T], protected var _label: Array[T])
    case class LocalLabeledImagePath(var label : Float, p : Path)
    class LabeledBGRImage(d: Array[Float], w: Int, h: Int, protected var _label : Float)
    class ArraySample[T: ClassTag](data: Array[T], featureSize: Array[Array[Int]], labelSize: Array[Array[Int]])
    ArrayTensorMiniBatch[T: ClassTag](val inputData: Array[Tensor[T]],val targetData: Array[Tensor[T]],
          featurePaddingParam: Option[PaddingParam[T]] = None, labelPaddingParam: Option[PaddingParam[T]] = None)
    BGRImage(data: Array[Float], protected var _width: Int, protected var _height: Int)
    case class LocalSeqFilePath(val path: Path)
    LabeledGreyImage(d: Array[Float], w: Int, h: Int, protected var _label : Float)
    case class SSDByteRecord(var data: Array[Byte], path: String)
    class RoiImage( val imInfo: Tensor[Float],  var target: RoiLabel = null)
    case class RoiLabel(classes: Tensor[Float], bboxes: Tensor[Float])
    class SSDMiniBatch(val input: Tensor[Float], val target: Tensor[Float], val imInfo: Tensor[Float] = null)
    RoiByteImage(var data: Array[Byte], var dataLength: Int, path: String, var target: RoiLabel = null)
    case class RoiImagePath(imagePath: String,target: RoiLabel = null) 

Existing transformer:
SampleToBatch           Transformer[Sample[T], MiniBatch[T]]                                        n:m
SampleToMiniBatch       Transformer[Sample[T], MiniBatch[T]]                                        n:m
GreyImgToBatch          Transformer[LabeledGreyImage, MiniBatch[Float]]                             n:m
LocalSeqFileToBytes     Transformer[LocalSeqFilePath, ByteRecord]                                   n:m
MTLabeledBGRImgToBatch  Transformer[A, MiniBatch[Float]]                                            n:m
BGRImgToLocalSeqFile    Transformer[(LabeledBGRImage, String), String]                              n:m
SampleToBatchNoPadding  Transformer[Sample[T], MiniBatch[T]]                                        n:m
BGRImgToBatch           Transformer[LabeledBGRImage, MiniBatch[Float]]                              n:m

img:
RowToByteRecords        Transformer[Row, ByteRecord]                                                1:1
toAutoencoderBatch      Transformer[MiniBatch[T], MiniBatch[T]]                                     1:1
LocalScaleImgWithName   Transformer[LocalLabeledImagePath, (LabeledBGRImage, String)]               1:1
BGRImgPixelNormalizer   Transformer[LabeledBGRImage, LabeledBGRImage]                               1:1
LocalScaleImgWithName   Transformer[LocalLabeledImagePath, (LabeledBGRImage, String)]               1:1
Identity                Transformer[A, A]                                                           1:1
BytesToBGRImg           Transformer[ByteRecord, LabeledBGRImage]                                    1:1
BGRImgRdmCropper        Transformer[LabeledBGRImage, LabeledBGRImage]                               1:1
ColorJitter             Transformer[LabeledBGRImage, LabeledBGRImage]                               1:1
BGRImgToSample          Transformer[LabeledBGRImage, Sample[Float]]                                 1:1
HFlip                   Transformer[LabeledBGRImage, LabeledBGRImage]                               1:1
LocalScaleImgReader     Transformer[LocalLabeledImagePath, LabeledBGRImage]                         1:1
BGRImgNormalizer        Transformer[LabeledBGRImage, LabeledBGRImage]                               1:1
GreyImgNormalizer       Transformer[LabeledGreyImage, LabeledGreyImage]                             1:1
Lighting                Transformer[LabeledBGRImage, LabeledBGRImage]                               1:1
BytesToGreyImg          Transformer[ByteRecord, LabeledGreyImage]                                   1:1
LocalResizeImgReader    Transformer[LocalLabeledImagePath, LabeledBGRImage]                         1:1
GreyImgToSample         Transformer[LabeledGreyImage, Sample[Float]]                                1:1
GreyImgCropper          Transformer[LabeledGreyImage, LabeledGreyImage]                             1:1
BGRImgCropper           Transformer[LabeledBGRImage, LabeledBGRImage]                               1:1
BGRImgToImageVector     Transformer[LabeledBGRImage, DenseVector]                                   1:1

SSD:
RoiByteImageToSeq       Transformer[RoiImagePath, String]                                           n:m
RoiImageToBatch         Transformer[RoiImage, SSDMiniBatch]                                         n:m
org.apache.spark.ml.image.DFInterface.Cropper                 Transformer[(RoiByteImage, Tensor[Float]), RoiByteImage]                    1:1
RecordToByteRoiImage    Transformer[SSDByteRecord, RoiByteImage]                                    1:1
RoiImageResizer         Transformer[RoiByteImage, RoiImage]                                         1:1
DataAugmentation        Transformer[RoiByteImage, RoiImage]                                         1:1
LocalByteRoiimageReader Transformer[RoiImagePath, SSDByteRecord]                                    1:1
RoiImageNormalizer      Transformer[RoiImage, RoiImage]                                             1:1
CVResizer               Transformer[RoiByteImage, RoiImage]                                         1:1

text:
TextToLabeledSentence   Transformer[Array[String], LabeledSentence[T]]                              1:1
SentenceTokenizer       Transformer[String, Array[String]]                                          1:1
SentenceBiPadding       Transformer[String, String]                                                 1:1
SentenceSplitter        Transformer[String, Array[String]]                                          1:1
LabeledSentenceToSample Transformer[LabeledSentence[T], Sample[T]]                                  1:1



need to support both RDD and DataFrame interface. Typical usage:

RDD:






training:
RDD[path] => RDD[(data, label)] => RDD[(data, label)] =>  RDD[MiniBatch] => Model

prediction
RDD[path] => RDD[(id, path, data, label)] => RDD[(id, path, data, label)] => RDD[(id, path, data, label, predict)] 


hard to use and extend with customer data
RDD[(a, b, c)] => RDD[(aa, b, c)], difficult to use transformer Iterator[a] => Iterator[aa] 







DataFrame
read file => tranform DataFrame => Array/Vector



















