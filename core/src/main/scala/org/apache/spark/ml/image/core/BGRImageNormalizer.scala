package org.apache.spark.ml.image.core

import java.util

import org.opencv.core.{Core, CvType, Mat, Scalar}

class BGRImageNormalizer (meanR: Float, meanG: Float, meanB: Float,
    stdR: Float = 1, stdG: Float = 1, stdB: Float = 1)

  extends ProcessStep[cMat, cMat] {

  override def apply(prev: cMat): cMat = {
    BGRImageNormalizer.transform(prev, meanR, meanG, meanB, stdR, stdG, stdB)
  }

}

object BGRImageNormalizer {
  def apply(meanR: Float, meanG: Float, meanB: Float,
            stdR: Float = 1, stdG: Float = 1, stdB: Float = 1): BGRImageNormalizer =
    new BGRImageNormalizer(meanR, meanG, meanB, stdR, stdG, stdB)

  def transform(input: cMat, meanR: Float, meanG: Float, meanB: Float,
                stdR: Float = 1, stdG: Float = 1, stdB: Float = 1): cMat = {
    if (input.`type`() != CvType.CV_32FC3) {
      input.convertTo(input, CvType.CV_32FC3)
    }
    val inputChannels = new util.ArrayList[Mat]()
    Core.split(input, inputChannels)
    require(inputChannels.size() == 3)
    val outputChannels = inputChannels

    Core.subtract(inputChannels.get(0), new Scalar(meanB), outputChannels.get(0))
    Core.subtract(inputChannels.get(1), new Scalar(meanG), outputChannels.get(1))
    Core.subtract(inputChannels.get(2), new Scalar(meanR), outputChannels.get(2))
    if (stdB != 1) Core.divide(outputChannels.get(0), new Scalar(stdB), outputChannels.get(0))
    if (stdG != 1) Core.divide(outputChannels.get(1), new Scalar(stdG), outputChannels.get(1))
    if (stdR != 1) Core.divide(outputChannels.get(2), new Scalar(stdR), outputChannels.get(2))
    Core.merge(outputChannels, input)

    (0 until inputChannels.size()).foreach(inputChannels.get(_).release())

    input
  }
}

