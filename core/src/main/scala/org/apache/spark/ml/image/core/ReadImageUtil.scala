package org.apache.spark.ml.image.core

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.InputStream
import javax.imageio.ImageIO

object ReadImageUtil extends Serializable {

  def readImageFromStream(
      inputStream: InputStream,
      scaleTo: Int,
      range: Float): (Array[Float], Array[Int]) ={
    val image = ImageIO.read(inputStream)

    val (widthAfterScale, heightAfterScale) =
      getWidthHeightAfterRatioScale(image.getHeight, image.getWidth, scaleTo)
    val bytes = resizeImage(image, widthAfterScale, heightAfterScale)

    val data = new Array[Float](bytes.length)
    var i = 0
    while (i < data.length) {
      data(i) = (bytes(i) & 0xff) / 255.0f * range
      i += 1
    }
    (data, Array(3, widthAfterScale, heightAfterScale))
  }

  private def getWidthHeightAfterRatioScale(
      oriHeight: Int,
      oriWidth: Int,
      scaleTo: Int): (Int, Int) = {
    if (oriWidth < oriHeight) {
      (scaleTo, scaleTo * oriHeight / oriWidth)
    } else {
      (scaleTo * oriWidth / oriHeight, scaleTo)
    }
  }

  private def resizeImage(img: BufferedImage, resizeWidth: Int, resizeHeight: Int): Array[Byte] = {
    val scaledImage: java.awt.Image =
      if ((resizeHeight == img.getHeight) && (resizeWidth == img.getWidth)) {
        img
      } else {
        img.getScaledInstance(resizeWidth, resizeHeight, java.awt.Image.SCALE_SMOOTH)
      }

    val imageBuff: BufferedImage =
      new BufferedImage(resizeWidth, resizeHeight, BufferedImage.TYPE_3BYTE_BGR)
    imageBuff.getGraphics.drawImage(scaledImage, 0, 0, new Color(0, 0, 0), null)
    val pixels: Array[Byte] =
      imageBuff.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    require(pixels.length % 3 == 0)
    pixels
  }


}
