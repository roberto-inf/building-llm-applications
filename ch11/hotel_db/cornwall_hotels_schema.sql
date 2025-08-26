-- Cornwall Hotels SQLite Schema and Data

CREATE TABLE IF NOT EXISTS hotels (
    hotel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel_name TEXT NOT NULL,
    town TEXT NOT NULL,
    address TEXT NOT NULL,
    rating REAL NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS hotel_room_offers (
    offer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel_id INTEGER NOT NULL,
    available_rooms INTEGER NOT NULL,
    price_single REAL NOT NULL,
    price_double REAL NOT NULL,
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
);

-- Pre-populate hotels
INSERT INTO hotels (hotel_name, town, address, rating, description) VALUES
('Seaview Hotel', 'Newquay', '1 Beach Rd, Newquay', 4.5, 'A beautiful hotel overlooking the sea.'),
('Harbour Inn', 'Falmouth', '12 Harbour St, Falmouth', 4.2, 'Charming inn near the harbour.'),
('Cornish Retreat', 'St Austell', '5 Retreat Ln, St Austell', 4.0, 'Relaxing retreat in the heart of Cornwall.'),
('Penzance Palace', 'Penzance', '22 Promenade, Penzance', 4.7, 'Luxury hotel with sea views.'),
('The Camborne Arms', 'Camborne', '8 Main St, Camborne', 4.1, 'Friendly hotel in Camborne.'),
('Hayle Haven', 'Hayle', '3 River Rd, Hayle', 4.3, 'Comfortable stay near the river.'),
('Land''s End Lodge', 'Land''s End', 'Land''s End Rd, Land''s End', 4.6, 'Stay at the edge of England.'),
('Bude Beach Hotel', 'Bude', '7 Beach Parade, Bude', 4.4, 'Steps from the sand.'),
('Padstow Quay Inn', 'Padstow', '2 Quay St, Padstow', 4.5, 'Quayside inn with great food.'),
('St Ives Bay Resort', 'St Ives', '9 Bay Rd, St Ives', 4.8, 'Resort with stunning bay views.');

-- Pre-populate hotel room offers
INSERT INTO hotel_room_offers (hotel_id, available_rooms, price_single, price_double) VALUES
(1, 5, 120.00, 180.00),
(2, 2, 95.00, 150.00),
(3, 8, 110.00, 170.00),
(4, 3, 130.00, 200.00),
(5, 4, 105.00, 160.00),
(6, 6, 99.00, 145.00),
(7, 2, 150.00, 220.00),
(8, 7, 115.00, 175.00),
(9, 5, 125.00, 185.00),
(10, 6, 140.00, 210.00); 