package org.apache.spark.ml.image.core

import java.util

import org.opencv.core.{Core, Mat}
import org.opencv.imgproc.Imgproc

/**
  * Created by yuhao on 9/21/17.
  */
class Hue (delta: Float, inplace: Boolean = true) extends ProcessStep[cMat, cMat] {
  override def apply(prev: cMat): cMat = {
    Brightness.transform(prev, prev, delta)
  }
}

object Hue {

  def apply(delta: Float, inplace: Boolean = true): Hue = new Hue(delta, inplace)

  def transform(inImg: cMat, outImg: cMat, delta: Float): cMat = {
    if (delta != 0) {
      // Convert to HSV colorspae
      Imgproc.cvtColor(inImg, outImg, Imgproc.COLOR_BGR2HSV)

      // Split the image to 3 channels.
      val channels = new util.ArrayList[Mat]()
      Core.split(outImg, channels)

      // Adjust the hue.
      channels.get(0).convertTo(channels.get(0), -1, 1, delta)
      Core.merge(channels, outImg)

      // Back to BGR colorspace.
      Imgproc.cvtColor(outImg, outImg, Imgproc.COLOR_HSV2BGR)
      outImg
    } else {
      inImg
    }
  }

}
