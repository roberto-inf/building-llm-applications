-- Create the UkBooking database schema

-- Create Destination table
CREATE TABLE Destination (
    DestinationId INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Description TEXT,
    Region TEXT,
    Country TEXT DEFAULT 'United Kingdom'
);

-- Create AccommodationType table
CREATE TABLE AccommodationType (
    AccommodationTypeId INTEGER PRIMARY KEY AUTOINCREMENT,
    TypeName TEXT NOT NULL
);

-- Create Accommodation table
CREATE TABLE Accommodation (
    AccommodationId INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Description TEXT,
    Address TEXT,
    DestinationId INTEGER,
    AccommodationTypeId INTEGER,
    FOREIGN KEY (DestinationId) REFERENCES Destination(DestinationId),
    FOREIGN KEY (AccommodationTypeId) REFERENCES AccommodationType(AccommodationTypeId)
);

-- Create Offer table
CREATE TABLE Offer (
    OfferId INTEGER PRIMARY KEY AUTOINCREMENT,
    AccommodationId INTEGER,
    OfferDescription TEXT,
    DiscountRate REAL,
    StartDate TEXT,
    EndDate TEXT,
    FOREIGN KEY (AccommodationId) REFERENCES Accommodation(AccommodationId)
);

-- Create Customer table
CREATE TABLE Customer (
    CustomerId INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT NOT NULL,
    LastName TEXT NOT NULL,
    Email TEXT UNIQUE,
    Phone TEXT,
    Address TEXT,
    City TEXT,
    PostalCode TEXT,
    Country TEXT DEFAULT 'United Kingdom'
);

-- Create Booking table
CREATE TABLE Booking (
    BookingId INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerId INTEGER,
    AccommodationId INTEGER,
    BookingDate TEXT,
    CheckInDate TEXT,
    CheckOutDate TEXT,
    TotalAmount REAL,
    FOREIGN KEY (CustomerId) REFERENCES Customer(CustomerId),
    FOREIGN KEY (AccommodationId) REFERENCES Accommodation(AccommodationId)
);
