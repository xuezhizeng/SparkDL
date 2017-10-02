package org.apache.spark.ml.image.core

import org.opencv.core.Core
import org.opencv.imgproc.Imgproc


class BGRToRGB extends ProcessStep[cMat, cMat] {

  override def apply(prev: cMat): cMat = {
    Imgproc.cvtColor(prev, prev, Imgproc.COLOR_BGR2RGB)
    prev
  }
}

object BGRToRGB {

  def apply(): BGRToRGB = new BGRToRGB()

}

