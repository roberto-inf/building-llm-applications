-- Populate Destination table
INSERT INTO Destination (Name, Description, Region, Country) VALUES
('London', 'Capital city of England', 'England', 'United Kingdom'),
('Edinburgh', 'Capital city of Scotland', 'Scotland', 'United Kingdom'),
('Cardiff', 'Capital city of Wales', 'Wales', 'United Kingdom'),
('Belfast', 'Capital city of Northern Ireland', 'Northern Ireland', 'United Kingdom'),
('Bath', 'City known for its Roman-built baths', 'England', 'United Kingdom'),
('York', 'Historic walled city in North Yorkshire', 'England', 'United Kingdom'),
('Liverpool', 'Maritime city in northwest England', 'England', 'United Kingdom'),
('Manchester', 'Major city in the northwest of England', 'England', 'United Kingdom'),
('Cambridge', 'University city in eastern England', 'England', 'United Kingdom'),
('Oxford', 'University city in central southern England', 'England', 'United Kingdom'),
('Bristol', 'City straddling the River Avon', 'England', 'United Kingdom'),
('Glasgow', 'Port city on the River Clyde in Scotland', 'Scotland', 'United Kingdom'),
('Inverness', 'Cultural capital of the Scottish Highlands', 'Scotland', 'United Kingdom'),
('Aberdeen', 'Port city in northeast Scotland', 'Scotland', 'United Kingdom'),
('Leeds', 'City in West Yorkshire, England', 'England', 'United Kingdom'),
('Newcastle', 'City on the River Tyne in northeast England', 'England', 'United Kingdom'),
('Nottingham', 'City in central England', 'England', 'United Kingdom'),
('Brighton', 'Seaside resort on the south coast of England', 'England', 'United Kingdom'),
('Canterbury', 'Historic English cathedral city', 'England', 'United Kingdom'),
('Plymouth', 'Port city on the south coast of Devon, England', 'England', 'United Kingdom');

-- Populate AccommodationType table
INSERT INTO AccommodationType (TypeName) VALUES
('Hotel'),
('Bed & Breakfast'),
('Camping'),
('Hostel'),
('Guest House'),
('Cottage'),
('Inn'),
('Resort'),
('Apartment'),
('Villa');

-- Populate Accommodation table
INSERT INTO Accommodation (Name, Description, Address, DestinationId, AccommodationTypeId) VALUES
('The Grand Hotel', 'Luxury hotel in the heart of London', '1 The Strand, London', 1, 1),
('Edinburgh B&B', 'Charming bed & breakfast with city views', '5 Royal Mile, Edinburgh', 2, 2),
('Cardiff Camping', 'Campsite near Cardiff Bay', 'Cardiff Bay, Cardiff', 3, 3),
('Belfast Hostel', 'Affordable hostel in central Belfast', '7 Donegall Square, Belfast', 4, 4),
('The Roman Baths Inn', 'Historic inn near the Roman Baths', '10 Bath St, Bath', 5, 7),
('York Guest House', 'Cozy guest house within the city walls', '12 York Way, York', 6, 5),
('Liverpool Cottage', 'Traditional cottage with modern amenities', '8 Liverpool Rd, Liverpool', 7, 6),
('Manchester Apartments', 'Serviced apartments in the city centre', '3 Canal St, Manchester', 8, 9),
('Cambridge Villa', 'Luxurious villa with garden views', '4 Kings Parade, Cambridge', 9, 10),
('Oxford Resort', 'Resort with spa and wellness center', '2 High St, Oxford', 10, 8),
('Bristol Inn', 'Quaint inn with easy access to the harbor', '6 Harbour St, Bristol', 11, 7),
('Glasgow Hotel', 'Modern hotel near the River Clyde', '11 Clyde St, Glasgow', 12, 1),
('Inverness Camping', 'Camping site with stunning highland views', '14 Ness Bank, Inverness', 13, 3),
('Aberdeen B&B', 'Cozy bed & breakfast near the coast', '9 Beach St, Aberdeen', 14, 2),
('Leeds Hostel', 'Hostel in the heart of Leeds', '15 City Square, Leeds', 15, 4),
('Newcastle Guest House', 'Elegant guest house near the river', '13 Quayside, Newcastle', 16, 5),
('Nottingham Hotel', 'Contemporary hotel in the city center', '16 Market Square, Nottingham', 17, 1),
('Brighton Villa', 'Villa with sea views in Brighton', '18 Marine Dr, Brighton', 18, 10),
('Canterbury Cottage', 'Charming cottage near the cathedral', '17 Cathedral Lane, Canterbury', 19, 6),
('Plymouth Resort', 'Seaside resort with private beach access', '19 Ocean Dr, Plymouth', 20, 8);

-- Populate Offer table
INSERT INTO Offer (AccommodationId, OfferDescription, DiscountRate, StartDate, EndDate) VALUES
(1, 'Summer Special', 0.15, '2024-06-01', '2024-08-31'),
(2, 'Weekend Getaway', 0.10, '2024-09-01', '2024-12-31'),
(3, 'Early Bird Discount', 0.20, '2024-05-01', '2024-06-30'),
(4, 'Stay 3 Nights, Get 1 Free', 0.25, '2024-01-01', '2024-03-31'),
(5, 'Historic Stay Offer', 0.10, '2024-04-01', '2024-06-30'),
(6, 'Autumn Discount', 0.15, '2024-09-01', '2024-11-30'),
(7, 'Cottage Retreat Offer', 0.12, '2024-07-01', '2024-09-30'),
(8, 'City Break Deal', 0.08, '2024-10-01', '2024-12-31'),
(9, 'Luxury Villa Offer', 0.18, '2024-05-01', '2024-08-31'),
(10, 'Spa & Wellness Package', 0.20, '2024-04-01', '2024-07-31');

-- Populate Customer table
INSERT INTO Customer (FirstName, LastName, Email, Phone, Address, City, PostalCode, Country) VALUES
('John', 'Doe', 'john.doe@example.com', '123456789', '123 Baker St', 'London', 'W1U 6TP', 'United Kingdom'),
('Jane', 'Smith', 'jane.smith@example.com', '987654321', '456 High St', 'Edinburgh', 'EH1 1PW', 'United Kingdom'),
('Alice', 'Johnson', 'alice.johnson@example.com', '555123456', '789 Park Rd', 'Cardiff', 'CF10 3AT', 'United Kingdom'),
('Robert', 'Brown', 'robert.brown@example.com', '444987654', '101 Greenway', 'Belfast', 'BT1 5GS', 'United Kingdom'),
('Emily', 'Davis', 'emily.davis@example.com', '222333444', '102 George St', 'Bath', 'BA1 2FS', 'United Kingdom'),
('Michael', 'Wilson', 'michael.wilson@example.com', '333444555', '103 York St', 'York', 'YO1 9RL', 'United Kingdom'),
('Sarah', 'Taylor', 'sarah.taylor@example.com', '666777888', '104 Lime St', 'Liverpool', 'L1 1JD', 'United Kingdom'),
('David', 'Lee', 'david.lee@example.com', '777888999', '105 Oxford Rd', 'Manchester', 'M1 5AN', 'United Kingdom'),
('Linda', 'Harris', 'linda.harris@example.com', '888999000', '106 Kings Parade', 'Cambridge', 'CB2 1ST', 'United Kingdom'),
('James', 'Clark', 'james.clark@example.com', '999000111', '107 High St', 'Oxford', 'OX1 4BH', 'United Kingdom'),
('Susan', 'Walker', 'susan.walker@example.com', '111222333', '108 Harbour St', 'Bristol', 'BS1 6XD', 'United Kingdom'),
('Charles', 'Allen', 'charles.allen@example.com', '333444555', '109 Clyde St', 'Glasgow', 'G1 2BB', 'United Kingdom'),
('Mary', 'Young', 'mary.young@example.com', '444555666', '110 Ness Bank', 'Inverness', 'IV2 4SF', 'United Kingdom'),
('Richard', 'King', 'richard.king@example.com', '555666777', '111 Beach St', 'Aberdeen', 'AB10 1HY', 'United Kingdom'),
('Patricia', 'Scott', 'patricia.scott@example.com', '666777888', '112 City Square', 'Leeds', 'LS1 5JB', 'United Kingdom'),
('Christopher', 'Evans', 'christopher.evans@example.com', '777888999', '113 Quayside', 'Newcastle', 'NE1 3AA', 'United Kingdom')