-- SQL Schema Dump
-- Database: kltn_stock_db
-- Generated on: 2026-06-04 09:58:01.715379

CREATE DATABASE IF NOT EXISTS `kltn_stock_db` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `kltn_stock_db`;

-- Table structure for table `daily_normalized_data`
DROP TABLE IF EXISTS `daily_normalized_data`;
CREATE TABLE `daily_normalized_data` (
  `ticker` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `date` date NOT NULL,
  `close_LogReturn` double DEFAULT NULL,
  `price_vs_sma50` double DEFAULT NULL,
  `volatility_20` double DEFAULT NULL,
  `volume_ratio_20` double DEFAULT NULL,
  `return_3d` double DEFAULT NULL,
  `return_5d` double DEFAULT NULL,
  `return_10d` double DEFAULT NULL,
  `return_20d` double DEFAULT NULL,
  `sma_50_LogReturn` double DEFAULT NULL,
  `volume_LogReturn` double DEFAULT NULL,
  `PCA_Trend` double DEFAULT NULL,
  `PCA_Oscillators` double DEFAULT NULL,
  `PCA_MACD` double DEFAULT NULL,
  `PCA_ShortReturns` double DEFAULT NULL,
  `atr_14` double DEFAULT NULL,
  `high_low` double DEFAULT NULL,
  `market_return` double DEFAULT NULL,
  `foreign_net` double DEFAULT NULL,
  `bu` double DEFAULT '0',
  `sd` double DEFAULT '0',
  `fs` double DEFAULT '0',
  `fb` double DEFAULT '0',
  `rs` double DEFAULT NULL,
  `rm` double DEFAULT NULL,
  PRIMARY KEY (`ticker`,`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table structure for table `daily_raw_data`
DROP TABLE IF EXISTS `daily_raw_data`;
CREATE TABLE `daily_raw_data` (
  `ticker` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `date` date NOT NULL,
  `open` double DEFAULT NULL,
  `high` double DEFAULT NULL,
  `low` double DEFAULT NULL,
  `close` double DEFAULT NULL,
  `volume` double DEFAULT NULL,
  PRIMARY KEY (`ticker`,`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table structure for table `model_predictions`
DROP TABLE IF EXISTS `model_predictions`;
CREATE TABLE `model_predictions` (
  `ticker` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `date` date NOT NULL,
  `prediction` int NOT NULL,
  `probability_up` double DEFAULT NULL,
  `probability_down` double DEFAULT NULL,
  `confidence` double DEFAULT NULL,
  `predict_date` date NOT NULL,
  PRIMARY KEY (`ticker`,`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table structure for table `model_training_data`
DROP TABLE IF EXISTS `model_training_data`;
CREATE TABLE `model_training_data` (
  `ticker` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `date` date NOT NULL,
  `close_LogReturn` double DEFAULT NULL,
  `price_vs_sma50` double DEFAULT NULL,
  `volatility_20` double DEFAULT NULL,
  `volume_ratio_20` double DEFAULT NULL,
  `return_3d` double DEFAULT NULL,
  `return_5d` double DEFAULT NULL,
  `return_10d` double DEFAULT NULL,
  `return_20d` double DEFAULT NULL,
  `sma_50_LogReturn` double DEFAULT NULL,
  `volume_LogReturn` double DEFAULT NULL,
  `PCA_Trend` double DEFAULT NULL,
  `PCA_Oscillators` double DEFAULT NULL,
  `PCA_MACD` double DEFAULT NULL,
  `PCA_ShortReturns` double DEFAULT NULL,
  `atr_14` double DEFAULT NULL,
  `high_low` double DEFAULT NULL,
  `market_return` double DEFAULT NULL,
  `foreign_net` double DEFAULT NULL,
  `target` int DEFAULT NULL,
  `bu` double DEFAULT '0',
  `sd` double DEFAULT '0',
  `fs` double DEFAULT '0',
  `fb` double DEFAULT '0',
  `rs` double DEFAULT NULL,
  `rm` double DEFAULT NULL,
  PRIMARY KEY (`ticker`,`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table structure for table `prediction_verification`
DROP TABLE IF EXISTS `prediction_verification`;
CREATE TABLE `prediction_verification` (
  `ticker` varchar(10) COLLATE utf8mb4_unicode_ci NOT NULL,
  `prediction_date` date NOT NULL,
  `predict_target_date` date NOT NULL,
  `prediction` int DEFAULT NULL,
  `probability_up` double DEFAULT NULL,
  `confidence` double DEFAULT NULL,
  `actual_outcome` int DEFAULT NULL,
  `is_correct` tinyint DEFAULT NULL,
  `actual_return` double DEFAULT NULL,
  `verified_at` datetime DEFAULT NULL,
  PRIMARY KEY (`ticker`,`prediction_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

