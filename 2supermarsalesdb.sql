-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Mar 13, 2025 at 01:35 PM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `2supermarsalesdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `feed`
--

CREATE TABLE `feed` (
  `id` bigint(250) NOT NULL auto_increment,
  `UserName` varchar(250) NOT NULL,
  `FeedBack` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=3 ;

--
-- Dumping data for table `feed`
--

INSERT INTO `feed` (`id`, `UserName`, `FeedBack`) VALUES
(1, 'fahad', 'good'),
(2, 'fahad', 'good one\r\n');

-- --------------------------------------------------------

--
-- Table structure for table `regtb`
--

CREATE TABLE `regtb` (
  `id` bigint(10) NOT NULL auto_increment,
  `username` varchar(250) NOT NULL,
  `mobile` varchar(250) NOT NULL,
  `email` varchar(250) NOT NULL,
  `password` varchar(250) NOT NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `regtb`
--

INSERT INTO `regtb` (`id`, `username`, `mobile`, `email`, `password`) VALUES
(1, 'mark', '9087254568', 'javaprojectfantasy@gmail.com', 'mark');
