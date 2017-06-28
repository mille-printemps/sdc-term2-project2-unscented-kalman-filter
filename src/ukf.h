#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "tools.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  // Initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // If this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // If this is false, radar measurements will be ignored (except for init)
  bool use_radar_;
  
  // State dimension
  int n_x_;
  
  // Measurement dimension
  int n_z_;
  
  // Augmented state dimension
  int n_aug_;
  
  // Sigma point spreading parameter
  double lambda_;
  
  // Weights of sigma points
  VectorXd weights_;

  // State vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  // State covariance matrix
  MatrixXd P_;
  
  // Mesurement matrix for lidar
  MatrixXd H_laser_;
  
  // Measurement covariance matrix for lidar
  MatrixXd R_laser_;
  
  // Measurement covariance matrix for radar
  MatrixXd R_radar_;

  // Predicted sigma points matrix
  MatrixXd Xsig_pred_;
  
  // Measurement sigma points matrix
  MatrixXd Zsig_;
  
  // Predicted measurement mean
  VectorXd z_pred_;
  
  // Predicted measurement covariance matrix
  MatrixXd S_;
  

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  long long previous_timestamp_;
  
  Tools tools_;


  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage &measurement_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Predict(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage &measurement_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage &measurement_package);
  
private:
  void AugmentSigmaPoints(MatrixXd &Xsig_out);
  void PredictSigmaPoints(double delta_t, MatrixXd &Xsig);
  void PredictMeanAndCovariance();
  void PredictRadarMeasurement();
  void UpdateRadarState(VectorXd &z);
  void UpdateLidarState(VectorXd &z);
};

#endif /* UKF_H */
