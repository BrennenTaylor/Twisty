#pragma once

#include <string>

namespace twisty
{
    class Curve;

    /**
     * @brief Responsible for writing out and loading curve save files in the twisty project.
     * Updates to this save functionality should be backwards compatable.
     *
     */
    class CurveWriter
    {
    public:
        /**
         * @brief Construct a new Curve Writer object
         *
         */
        CurveWriter();

        /**
         * @brief Resets the referenced values in the writer. A curve file written at this point would be empty.
         *
         */
        void BeginSettingValues();
        /**
         * @brief Writes out the values saved in the current curve writer.
         *
         * @param filename Name of the file to write out the values to
         */
        void EndSettingValuesAndWrite(std::string filename);

        /**
         * @brief Set the Curve object
         *
         * @param pCurve Curve to write out
         */
        void SetCurve(Curve* pCurve);

    private:
        Curve* m_pCurve;

    public:
        // static load values
        static const uint32_t CurveIdValue;
        static const uint32_t BezierInfoIdValue;
        static const uint32_t GtPositionsIdValue;
        static const uint32_t GtFramesIdValue;
        static const std::string CurveDataFileExt;
    };
}