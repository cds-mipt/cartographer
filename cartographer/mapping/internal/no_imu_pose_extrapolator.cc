/*
 * Copyright 2017 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/no_imu_pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

NoImuPoseExtrapolator::NoImuPoseExtrapolator(const common::Duration pose_queue_duration)
    : pose_queue_duration_(pose_queue_duration),
      cached_extrapolated_pose_{common::Time::min(),
                                transform::Rigid3d::Identity()} {}

common::Time NoImuPoseExtrapolator::GetLastPoseTime() const {
  if (timed_pose_queue_.empty()) {
    return common::Time::min();
  }
  return timed_pose_queue_.back().time;
}

common::Time NoImuPoseExtrapolator::GetLastExtrapolatedTime() const {
  return last_extrapolated_time_;
}

void NoImuPoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose) {
  timed_pose_queue_.push_back(TimedPose{time, pose});
  while (timed_pose_queue_.size() > 2 &&
         timed_pose_queue_[1].time <= time - pose_queue_duration_) {
    timed_pose_queue_.pop_front();
  }
  UpdateVelocitiesFromPoses();
  TrimOdometryData();
}

void NoImuPoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  CHECK(timed_pose_queue_.empty() ||
        odometry_data.time >= timed_pose_queue_.back().time);
  odometry_data_.push_back(odometry_data);
  TrimOdometryData();
  if (odometry_data_.size() < 2) {
    return;
  }
  // TODO(whess): Improve by using more than just the last two odometry poses.
  // Compute extrapolation in the tracking frame.
  const sensor::OdometryData& odometry_data_oldest = odometry_data_.front();
  const sensor::OdometryData& odometry_data_newest = odometry_data_.back();
  const double odometry_time_delta =
      common::ToSeconds(odometry_data_oldest.time - odometry_data_newest.time);
  const transform::Rigid3d odometry_pose_delta =
      odometry_data_newest.pose.inverse() * odometry_data_oldest.pose;
  angular_velocity_from_odometry_ =
      transform::RotationQuaternionToAngleAxisVector(
          odometry_pose_delta.rotation()) /
      odometry_time_delta;
  if (timed_pose_queue_.empty()) {
    return;
  }
  const Eigen::Vector3d
      linear_velocity_in_tracking_frame_at_newest_odometry_time =
          odometry_pose_delta.translation() / odometry_time_delta;
  const Eigen::Quaterniond orientation_at_newest_odometry_time =
      timed_pose_queue_.back().pose.rotation() *
      ExtrapolateRotation(odometry_data_newest.time);
  linear_velocity_from_odometry_ =
      orientation_at_newest_odometry_time *
      linear_velocity_in_tracking_frame_at_newest_odometry_time;
}

transform::Rigid3d NoImuPoseExtrapolator::ExtrapolatePose(common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  if (cached_extrapolated_pose_.time != time) {
    const Eigen::Vector3d translation =
        ExtrapolateTranslation(time) + newest_timed_pose.pose.translation();
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() * ExtrapolateRotation(time);
    cached_extrapolated_pose_ =
        TimedPose{time, transform::Rigid3d{translation, rotation}};
  }
  return cached_extrapolated_pose_.pose;
}

Eigen::Quaterniond NoImuPoseExtrapolator::EstimateGravityOrientation(common::Time time) {
  return Eigen::Quaterniond(1, 0, 0, 0);  // w, x, y, z
}

void NoImuPoseExtrapolator::UpdateVelocitiesFromPoses() {
  if (timed_pose_queue_.size() < 2) {
    // We need two poses to estimate velocities.
    return;
  }
  CHECK(!timed_pose_queue_.empty());
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const auto newest_time = newest_timed_pose.time;
  const TimedPose& oldest_timed_pose = timed_pose_queue_.front();
  const auto oldest_time = oldest_timed_pose.time;
  const double queue_delta = common::ToSeconds(newest_time - oldest_time);
  if (queue_delta < common::ToSeconds(pose_queue_duration_)) {
    LOG(WARNING) << "Queue too short for velocity estimation. Queue duration: "
                 << queue_delta << " s";
    return;
  }
  const transform::Rigid3d& newest_pose = newest_timed_pose.pose;
  const transform::Rigid3d& oldest_pose = oldest_timed_pose.pose;
  linear_velocity_from_poses_ =
      (newest_pose.translation() - oldest_pose.translation()) / queue_delta;
  angular_velocity_from_poses_ =
      transform::RotationQuaternionToAngleAxisVector(
          oldest_pose.rotation().inverse() * newest_pose.rotation()) /
      queue_delta;
}

void NoImuPoseExtrapolator::TrimOdometryData() {
  while (odometry_data_.size() > 2 && !timed_pose_queue_.empty() &&
         odometry_data_[1].time <= timed_pose_queue_.back().time) {
    odometry_data_.pop_front();
  }
}

Eigen::Quaterniond NoImuPoseExtrapolator::ExtrapolateRotation(common::Time time) {
  CHECK_GE(time, last_extrapolated_time_);
  last_extrapolated_time_ = time;
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const double extrapolation_delta =
      common::ToSeconds(time - newest_timed_pose.time);
  if (odometry_data_.size() < 2) {
    return Eigen::Quaterniond(
        Eigen::AngleAxisd(extrapolation_delta * angular_velocity_from_poses_.norm(), angular_velocity_from_poses_.normalized()));
  }
  return Eigen::Quaterniond(
      Eigen::AngleAxisd(extrapolation_delta * angular_velocity_from_odometry_.norm(), angular_velocity_from_odometry_.normalized()));
}

Eigen::Vector3d NoImuPoseExtrapolator::ExtrapolateTranslation(common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const double extrapolation_delta =
      common::ToSeconds(time - newest_timed_pose.time);
  if (odometry_data_.size() < 2) {
    return extrapolation_delta * linear_velocity_from_poses_;
  }
  return extrapolation_delta * linear_velocity_from_odometry_;
}

NoImuPoseExtrapolator::ExtrapolationResult
NoImuPoseExtrapolator::ExtrapolatePosesWithGravity(
    const std::vector<common::Time>& times) {
  std::vector<transform::Rigid3f> poses;
  for (auto it = times.begin(); it != std::prev(times.end()); ++it) {
    poses.push_back(ExtrapolatePose(*it).cast<float>());
  }

  const Eigen::Vector3d current_velocity = odometry_data_.size() < 2
                                               ? linear_velocity_from_poses_
                                               : linear_velocity_from_odometry_;
  return ExtrapolationResult{poses, ExtrapolatePose(times.back()),
                             current_velocity,
                             EstimateGravityOrientation(times.back())};
}

}  // namespace mapping
}  // namespace cartographer
