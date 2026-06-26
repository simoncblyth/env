CREATE TABLE device_telemetry (
    t TEXT NOT NULL,
    hostname TEXT NOT NULL,
    cpu_utilization REAL,
    memory_utilization REAL
);

-- Insert sample metrics spaced by 5-minute intervals
INSERT INTO device_telemetry VALUES (datetime('now', '-20 minutes'), 'node-01', 42.5, 68.2);
INSERT INTO device_telemetry VALUES (datetime('now', '-15 minutes'), 'node-01', 48.1, 68.4);
INSERT INTO device_telemetry VALUES (datetime('now', '-10 minutes'), 'node-01', 55.0, 71.0);
INSERT INTO device_telemetry VALUES (datetime('now', '-5 minutes'),  'node-01', 63.8, 72.1);
INSERT INTO device_telemetry VALUES (datetime('now'),                'node-01', 51.2, 70.5);

INSERT INTO device_telemetry VALUES (datetime('now', '-20 minutes'), 'node-02', 12.1, 34.0);
INSERT INTO device_telemetry VALUES (datetime('now', '-15 minutes'), 'node-02', 14.5, 34.1);
INSERT INTO device_telemetry VALUES (datetime('now', '-10 minutes'), 'node-02', 11.2, 33.9);
INSERT INTO device_telemetry VALUES (datetime('now', '-5 minutes'),  'node-02', 18.9, 35.2);
INSERT INTO device_telemetry VALUES (datetime('now'),                'node-02', 15.4, 34.8);

