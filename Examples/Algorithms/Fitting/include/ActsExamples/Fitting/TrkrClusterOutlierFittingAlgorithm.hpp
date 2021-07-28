#pragma once

#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <boost/program_options.hpp>

#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/MagneticField/InterpolatedBFieldMap.hpp"
#include "Acts/MagneticField/SharedBField.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/ParameterDefinitions.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"

#include "ActsExamples/Plugins/BField/ScalableBField.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/BareAlgorithm.hpp"
#include "ActsExamples/Plugins/BField/BFieldOptions.hpp"
#include "ActsExamples/EventData/TrkrClusterSourceLink.hpp"

struct ResidualOutlierFinder {
  long unsigned int tpcID;
  double residualCut;
  double chi2Cut;

  template <typename track_state_t>
  bool operator()(const track_state_t& state) const {
    
    /// can't determine an outlier without a measusrement or prediction from KF
    if(not state.hasCalibrated() or not state.hasPredicted())
      return false;

    auto residuals = state.calibrated() - state.projector() * state.predicted();

    /// Get the relevant state parameters
    const auto& projector = state.projector();

    const auto& predictedCov = state.predictedCovariance();
    const auto& stateCov = state.effectiveCalibratedCovariance();

    double chi2 = (residuals.transpose() *
		   ((stateCov + projector * predictedCov * projector.transpose())).inverse() * residuals).eval()(0,0);

    std::cout << "Chi2 is " << chi2<<std::endl;

    auto distance = residuals.norm();
    std::cout << "Distance is " << distance << std::endl;

    //auto volID = state.referenceSurface().geometryId().volume();
    //double distanceMax = std::numeric_limits<double>::max();
 
    return (chi2 <= chi2Cut);
  }
};


namespace ActsExamples {

/**
 * This class contains the information required to run the Kalman fitter
 * with the TrkrClusterSourceLinks. Based on ActsExamples::FittingAlgorithm
 */
class TrkrClusterOutlierFittingAlgorithm : public BareAlgorithm
{
 public:
  /// Construct some aliases to be used for the fitting results
  using FitterResult
    = Acts::Result<Acts::KalmanFitterResult<ActsExamples::TrkrClusterSourceLink>>;
  using FitterFunction
    = std::function<FitterResult(
		    const std::vector<ActsExamples::TrkrClusterSourceLink>&,
		    const TrackParameters&,
		    const Acts::KalmanFitterOptions<ResidualOutlierFinder>&)>;

  using DirectedFitterFunction
    = std::function<FitterResult(
		    const std::vector<ActsExamples::TrkrClusterSourceLink>&,
		    const TrackParameters&,
		    const Acts::KalmanFitterOptions<ResidualOutlierFinder>&,
		    const std::vector<const Acts::Surface*>&)>;
      

  /// Create fitter function
  static FitterFunction makeFitterFunction(
      std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
      Options::BFieldVariant magneticField);

  static DirectedFitterFunction makeFitterFunction(
      Options::BFieldVariant magneticField);

  struct Config
  {
    FitterFunction fit;
    DirectedFitterFunction dFit;
  };

  /// Constructor 
  TrkrClusterOutlierFittingAlgorithm(Config cfg, Acts::Logging::Level lvl);


private:
  Config m_cfg;
};

}
