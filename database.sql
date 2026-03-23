-- ============================================================
--  SkinSense — Full Database Setup
--  Open phpMyAdmin → Import → select this file → click GO
-- ============================================================

CREATE DATABASE IF NOT EXISTS skin_db;
USE skin_db;

-- ── USERS ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(100)  NOT NULL,
    email      VARCHAR(100)  NOT NULL UNIQUE,
    password   VARCHAR(255)  NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── DOCTORS ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS doctors (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(100) NOT NULL,
    specialization VARCHAR(100) NOT NULL,
    disease_class  VARCHAR(100) NOT NULL,
    contact        VARCHAR(20)  NOT NULL
);

-- ── FEEDBACK ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT  NOT NULL,
    message    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ── SAMPLE DOCTORS DATA ───────────────────────────────────
INSERT INTO doctors (name, specialization, disease_class, contact) VALUES
('Dr. Arjun Mehta',   'Dermatologist',  'Acne',                 '9876543210'),
('Dr. Priya Nair',    'Dermatologist',  'Acne',                 '9123456780'),
('Dr. Sunita Sharma', 'Dermatologist',  'Rosacea',              '9345678901'),
('Dr. Kiran Reddy',   'Skin Specialist','Rosacea',              '9456123780'),
('Dr. Karan Singh',   'Skin Specialist','Actinic Keratosis',    '9456789012'),
('Dr. Meena Patel',   'Dermatologist',  'Actinic Keratosis',    '9567890123'),
('Dr. Ravi Kumar',    'Oncologist',     'Basal Cell Carcinoma', '9234567890'),
('Dr. Anjali Desai',  'Oncologist',     'Basal Cell Carcinoma', '9876012345'),
('Dr. Priya Nair',    'Oncologist',     'Melanoma',             '9123456789'),
('Dr. Suresh Menon',  'Oncologist',     'Melanoma',             '9012345678'),
('Dr. Meena Patel',   'Dermatologist',  'Dysplastic Nevi',      '9567890123'),
('Dr. Anand Rao',     'Skin Specialist','Moles',                '9678901234'),
('Dr. Deepa Verma',   'Oncologist',     'Malignant Lesions',    '9789012345'),
('Dr. Rajesh Iyer',   'Oncologist',     'Malignant Lesions',    '9890123456');