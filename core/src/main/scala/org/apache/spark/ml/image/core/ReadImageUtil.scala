package org.apache.spark.ml.image.core

import com.intel.analytics.OpenCV
import org.opencv.core.{MatOfByte, Size}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

object ReadImageUtil extends Serializable {


  OpenCV.loadAsNecessary()

  /**
   * @param fileContent bytes representation for an image file.
   * @return OpenCV Mat
   */
  def readImageAsMat(fileContent: Array[Byte]): cMat = {
    val mat = Imgcodecs.imdecode(new MatOfByte(fileContent: _*), Imgcodecs.CV_LOAD_IMAGE_COLOR)
    new cMat(mat)
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @param smallSideSize the size of the smallest side after resize
   * @return OpenCV Mat
   */
  def readImageAsMat(fileContent: Array[Byte], smallSideSize: Int): cMat = {
    aspectPreseveRescale(readImageAsMat(fileContent), smallSideSize)
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @return (data, height, width, numChannel)
   */
  def readImageAsBytes(fileContent: Array[Byte]): (Array[Byte], Int, Int, Int) ={
    val mat = readImageAsMat(fileContent)
    mat2Bytes(mat)
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @param smallSideSize the size of the smallest side after resize
   * @return (data, height, width, numChannel)
   */
  def readImageAsBytes(fileContent: Array[Byte], smallSideSize: Int): (Array[Byte], Int, Int, Int) = {
    val mat = aspectPreseveRescale(readImageAsMat(fileContent), smallSideSize)
    mat2Bytes(mat)
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @param smallSideSize the size of the smallest side after resize
   * @param divisor divide each pixel by divisor. E.g. if divisor = 255, each pixel is in [0, 1]
   * @return (data, height, width, numChannel)
   */
  def readImageAsFloats(
      fileContent: Array[Byte],
      smallSideSize: Int,
      divisor: Float = 1.0f): (Array[Float], Int, Int, Int) = {
    val (bytes, h, w, c) = readImageAsBytes(fileContent, smallSideSize)
    val floats = bytes.map(b => (b & 0xff) / divisor)
    (floats, h, w, c)
  }

  private def mat2Bytes(mat: cMat): (Array[Byte], Int, Int, Int) = {
    val w = mat.width()
    val h = mat.height()
    val c = mat.channels()
    val bytes = Array.ofDim[Byte](c * w * h)
    mat.get(0, 0, bytes)
    (bytes, h, w, c)
  }

  private def aspectPreseveRescale(srcMat: cMat, smallSideSize: Int): cMat = {
    val origW = srcMat.width()
    val origH = srcMat.height()
    val (resizeW, resizeH) = if (origW < origH) {
      (smallSideSize, origH * smallSideSize  / origW)
    } else {
      (origW * smallSideSize / origH, smallSideSize)
    }
    val dst = srcMat
    Imgproc.resize(srcMat, dst, new Size(resizeW, resizeH), 0, 0, Imgproc.INTER_LINEAR)
    dst
  }
}


