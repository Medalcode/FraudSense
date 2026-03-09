-- ============================================================
-- FraudSense — Schema de Base de Datos
-- Sistema Inteligente de Detección de Fraude
-- ============================================================

-- Tabla principal de transacciones financieras
CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_date    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    amount              DECIMAL(15, 2)  NOT NULL,       -- Monto en CLP
    country             VARCHAR(3)      NOT NULL,       -- Código ISO país (CL, AR, RU...)
    hour                TINYINT         NOT NULL,       -- Hora del día (0-23)
    device_type         VARCHAR(20)     NOT NULL,       -- Android, iOS, Web, Windows, Unknown
    failed_attempts     TINYINT         DEFAULT 0,      -- Intentos fallidos previos
    is_foreign          BOOLEAN         DEFAULT FALSE,  -- ¿País diferente al habitual?
    high_risk_merchant  BOOLEAN         DEFAULT FALSE,  -- ¿Comercio de alto riesgo?
    is_fraud            BOOLEAN         DEFAULT NULL,   -- Etiqueta real (para histórico)
    created_at          TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de clientes
CREATE TABLE IF NOT EXISTS clients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            VARCHAR(100)    NOT NULL,
    email           VARCHAR(150)    UNIQUE NOT NULL,
    home_country    VARCHAR(3)      NOT NULL DEFAULT 'CL',
    usual_devices   VARCHAR(200),                       -- JSON array de dispositivos habituales
    avg_amount      DECIMAL(15, 2)  DEFAULT 0,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de alertas de fraude generadas por el modelo
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  INTEGER         NOT NULL REFERENCES transactions(id),
    risk_score      DECIMAL(5, 4)   NOT NULL,           -- Probabilidad de fraude (0–1)
    risk_level      VARCHAR(10)     NOT NULL,           -- BAJO / MEDIO / ALTO
    recommendation  VARCHAR(100)    NOT NULL,           -- Acción recomendada
    reviewed_by     VARCHAR(100),                       -- Analista que revisó (si aplica)
    reviewed_at     TIMESTAMP,                          -- Fecha de revisión
    final_verdict   VARCHAR(20),                        -- FRAUDE / LEGÍTIMA / PENDIENTE
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de puntuaciones de riesgo históricas del modelo
CREATE TABLE IF NOT EXISTS risk_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  INTEGER         NOT NULL REFERENCES transactions(id),
    model_version   VARCHAR(20)     NOT NULL DEFAULT '1.0.0',
    score           DECIMAL(5, 4)   NOT NULL,
    features_used   TEXT,                               -- JSON con features relevantes
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Índices para optimizar consultas frecuentes
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_transactions_date      ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_country   ON transactions(country);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud  ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_score     ON fraud_alerts(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_level     ON fraud_alerts(risk_level);

-- ============================================================
-- Vistas útiles para el Dashboard
-- ============================================================

-- Vista: Resumen diario de fraude
CREATE VIEW IF NOT EXISTS daily_fraud_summary AS
SELECT
    DATE(transaction_date)  AS day,
    COUNT(*)                AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2) AS fraud_rate_pct,
    AVG(amount)             AS avg_amount,
    MAX(amount)             AS max_amount
FROM transactions
GROUP BY DATE(transaction_date);

-- Vista: Fraude por país
CREATE VIEW IF NOT EXISTS fraud_by_country AS
SELECT
    country,
    COUNT(*)                AS total,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount ELSE NULL END), 0) AS avg_fraud_amount
FROM transactions
GROUP BY country
ORDER BY fraud_count DESC;
