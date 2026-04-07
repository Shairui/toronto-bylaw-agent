"""RAG (Retrieval-Augmented Generation) knowledge base for Toronto bylaws."""
import json
import os
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from backend.config import CHROMA_DB_PATH, KNOWLEDGE_BASE_PATH


class TorontoBylawRAG:
    """RAG system for Toronto bylaw knowledge base."""

    def __init__(self):
        """Initialize RAG system with ChromaDB."""
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="toronto_bylaws",
            metadata={"hnsw:space": "cosine"}
        )
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load knowledge base from JSON file."""
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                return json.load(f)
        return self._create_default_knowledge_base()

    def _create_default_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create Toronto bylaw knowledge base with 55+ documents."""
        return [
            # ── ZONING ──────────────────────────────────────────────────────
            {
                "id": "bylaw_001",
                "title": "Toronto Zoning By-law 569-2013 – Overview",
                "content": "Zoning By-law 569-2013 is the comprehensive zoning by-law for the City of Toronto. It regulates how land can be used and sets rules for building size, height, and placement. It replaces the former six district zoning by-laws. The by-law divides Toronto into residential, commercial, employment, and mixed-use zones. Each zone has rules for permitted uses, density, setbacks, lot coverage, building heights, and parking.",
                "source": "Toronto Zoning By-law 569-2013",
                "category": "Zoning"
            },
            {
                "id": "bylaw_002",
                "title": "Residential (R) Zones – Detached Housing",
                "content": "The R zone (Residential) permits detached houses and semi-detached houses. Sub-categories R1 through R4 allow progressively denser forms. R1 permits single-detached houses only. R2 permits detached and semi-detached. R3 permits detached, semi-detached, and townhouses. R4 permits up to 4-unit townhouses. Typical front yard setback is 6 m. Lot coverage limit is generally 33%. Maximum height for residential zones is typically 10 m (about 2.5 storeys).",
                "source": "Toronto Zoning By-law 569-2013, Section 20",
                "category": "Zoning"
            },
            {
                "id": "bylaw_003",
                "title": "Residential Apartment (RA) Zones",
                "content": "RA zones permit apartment buildings as the primary use. Building heights, density (FSI – Floor Space Index), and setbacks vary by sub-zone (RA1–RA3). RA1 typically allows up to 8 storeys. RA2 allows up to 20 storeys. RA3 has no height limit but requires site-plan approval. Underground or structured parking is often required for buildings over 6 storeys. Amenity space of at least 4 m² per unit is required.",
                "source": "Toronto Zoning By-law 569-2013, Section 25",
                "category": "Zoning"
            },
            {
                "id": "bylaw_004",
                "title": "Townhouse and Row Housing (RT) Zones",
                "content": "RT zones permit townhouses and row houses on individual lots or as part of a complex. Maximum building height is usually 12 m. A private outdoor amenity area of at least 15 m² per unit is required. Parking must be provided: one space per unit for buildings under 6 units. RT zones allow fee-simple ownership (freehold townhouses) or condominium tenure.",
                "source": "Toronto Zoning By-law 569-2013, Section 24",
                "category": "Zoning"
            },
            {
                "id": "bylaw_005",
                "title": "Mixed-Use (CR) Zones – Commercial-Residential",
                "content": "CR (Commercial-Residential) zones permit a mix of residential and non-residential uses. The zone uses two numbers: CR 2.0 (c1.0; r1.5) means total FSI of 2.0, commercial FSI of 1.0, and residential FSI of 1.5. Ground floors of CR buildings often must contain active retail or commercial uses. Residential units above ground floor are permitted. CR zones are typically found along main streets such as Bloor, Queen, and Danforth.",
                "source": "Toronto Zoning By-law 569-2013, Section 30",
                "category": "Zoning"
            },
            {
                "id": "bylaw_006",
                "title": "Commercial (C) Zones",
                "content": "Commercial zones (C1–C4) permit retail, office, service, and hospitality uses. C1 is local commercial (neighbourhood stores). C2 is general commercial (larger format retail). C3 is heavy commercial (auto-oriented). C4 is special commercial (entertainment, sports). Drive-throughs are restricted in many C zones and require a minimum queue of 6 vehicles. Most C zones prohibit new stand-alone residential uses.",
                "source": "Toronto Zoning By-law 569-2013, Section 35",
                "category": "Zoning"
            },
            {
                "id": "bylaw_007",
                "title": "Employment (E) Zones",
                "content": "Employment zones (E1–E3) are intended for manufacturing, warehousing, utilities, and compatible service uses. E1 (Core Employment) protects industrial land from residential conversion. E2 (General Employment) allows more flexible uses. E3 (Employment District) allows limited retail. Residential uses are prohibited in E1 and E2 zones to protect employment land supply. Minimum lot area in E zones is typically 1,000 m².",
                "source": "Toronto Zoning By-law 569-2013, Section 40",
                "category": "Zoning"
            },
            {
                "id": "bylaw_008",
                "title": "Setback Requirements",
                "content": "Setbacks are the required distances between a building and a lot line. Front yard setback for residential zones: typically 6 m (may match the average of adjacent buildings in some areas). Rear yard setback: 7.5 m for residential. Side yard setback: typically 0.9 m each side for detached houses; 0 m for the shared wall of semi-detached. Additions to existing non-conforming buildings may be exempt from setback rules if they do not increase non-conformity.",
                "source": "Toronto Zoning By-law 569-2013, Section 4",
                "category": "Zoning"
            },
            {
                "id": "bylaw_009",
                "title": "Height Restrictions by Zone",
                "content": "Building height limits vary by zone: Residential R1-R4: 10 m. RT (Townhouse): 12 m. RA1: 22 m. RA2: 58 m. CR zones: heights depend on suffix (e.g., CR 3.0 (c2.0; r2.5) height 20 m). Heights may be further limited by angular plane rules (45-degree angular plane from the rear lot line). Heritage overlays or view corridor protections may impose additional limits, particularly near Lake Ontario or significant heritage landmarks.",
                "source": "Toronto Zoning By-law 569-2013, Section 4",
                "category": "Zoning"
            },
            {
                "id": "bylaw_010",
                "title": "Secondary Suites and Garden Suites",
                "content": "Secondary suites (basement apartments or in-law suites) are permitted as-of-right in all residential zones across Toronto following Bill 23 (2022). A secondary suite must have a separate entrance, a kitchen, a bathroom, and a sleeping area. Garden suites (separate small dwellings in the backyard) are also now permitted in most residential zones. Garden suites may be up to 60 m² (or 16% of the lot area) and a maximum of 6 m in height. A building permit is required for both types.",
                "source": "Toronto Zoning By-law 569-2013, Section 150; City of Toronto Garden Suite Guidelines 2022",
                "category": "Zoning"
            },
            {
                "id": "bylaw_011",
                "title": "Parking Requirements by Zone",
                "content": "Minimum parking spaces: Detached/semi-detached house: 1 space per unit. Townhouse: 1 space per unit. Apartment: varies 0.5–1.0 per unit depending on proximity to subway. Office: 1 space per 50–100 m² of GFA. Retail: 1 space per 30–40 m² of GFA. In the Downtown Core (bounded by Bathurst, Bloor, the DVP, and the lakefront) residential minimums are waived. Bicycle parking is required for all new multi-residential buildings (1 long-term space per unit).",
                "source": "Toronto Zoning By-law 569-2013, Section 200",
                "category": "Zoning"
            },
            {
                "id": "bylaw_012",
                "title": "Short-Term Rentals Zoning Regulations",
                "content": "Short-term rentals (e.g., Airbnb) are permitted in Toronto only in a host's principal residence. Entire-home rentals are allowed up to 180 nights per calendar year. Room rentals within the principal residence are unlimited. Short-term rental operators must register with the City and pay the Municipal Accommodation Tax (4% of the listing price). Short-term rentals are not permitted in non-principal residences (e.g., investment condos) or in Employment zones.",
                "source": "Toronto Municipal Code Chapter 547 (Short-Term Rentals)",
                "category": "Zoning"
            },
            {
                "id": "bylaw_013",
                "title": "Greenbelt and Natural Heritage Areas",
                "content": "Areas designated as Natural Areas, Ravines, or Environmental Protection (EP) zones are protected from development. The Ravine and Natural Feature Protection By-law (By-law 2-2017) prohibits injuring or destroying trees, grading, or dumping within Toronto's ravine system. A Ravine Special Permit is required for any work within 10 m of the ravine edge. Development near the Toronto and Region Conservation Authority (TRCA) regulated areas also requires TRCA approval.",
                "source": "Toronto Ravine and Natural Feature Protection By-law 2-2017; Toronto Zoning By-law 569-2013",
                "category": "Zoning"
            },
            {
                "id": "bylaw_014",
                "title": "Lot Coverage Rules",
                "content": "Lot coverage is the percentage of a lot covered by all buildings (house, garage, shed, etc.). In R1 and R2 zones: maximum 33% lot coverage. In R3 and R4 zones: up to 40% lot coverage. Detached garages and sheds under 10 m² do not require a building permit but still count toward lot coverage. Pools, patios, and driveways do not count toward lot coverage. Exceeding lot coverage requires a minor variance from the Committee of Adjustment.",
                "source": "Toronto Zoning By-law 569-2013, Section 4",
                "category": "Zoning"
            },
            {
                "id": "bylaw_015",
                "title": "Minor Variances and Committee of Adjustment",
                "content": "If a project does not comply with zoning by-law standards, the owner may apply for a minor variance from the Committee of Adjustment. The Committee may approve a variance if it meets four tests: it is minor in nature, it is desirable for the appropriate development of the land, it maintains the general intent of the Official Plan, and it maintains the general intent of the Zoning By-law. Application fee: approximately $1,400–$2,100 depending on project type. Decisions can be appealed to the Ontario Land Tribunal.",
                "source": "Planning Act, R.S.O. 1990; Toronto Committee of Adjustment",
                "category": "Zoning"
            },

            # ── BUILDING PERMITS ─────────────────────────────────────────────
            {
                "id": "bylaw_016",
                "title": "When a Building Permit Is Required",
                "content": "Building permits are required for: new construction of any building over 10 m²; additions to existing buildings; demolition or relocation of a building; change of use of a building; structural alterations; installation or alteration of HVAC systems; new or altered plumbing, drains, or water services; construction of a deck more than 600 mm above grade; installation of a new pool or hot tub; installation or alteration of a fireplace or woodstove. Exemptions: painting, decorating, flooring, cabinet installation, minor repairs that do not affect structural elements.",
                "source": "Toronto Municipal Code Chapter 363 (Buildings); Ontario Building Code",
                "category": "Permits"
            },
            {
                "id": "bylaw_017",
                "title": "How to Apply for a Building Permit",
                "content": "Building permit applications are submitted through the Toronto Building portal at toronto.ca/building. Required documents: completed application form; site plan showing the location of all buildings on the lot; construction drawings (floor plans, elevations, sections) signed and sealed by an architect or engineer (required for complex projects); survey or lot grading plan. Simple residential projects (e.g., decks, small additions) may not require sealed drawings. Review times vary: small residential projects 10–20 business days; larger projects 30–60 business days.",
                "source": "Toronto Municipal Code Chapter 363; toronto.ca/building",
                "category": "Permits"
            },
            {
                "id": "bylaw_018",
                "title": "Building Permit Fees",
                "content": "Building permit fees are based on the value of construction. Residential: $13.73 per $1,000 of construction value (2024 rate). Commercial/Industrial: approximately $18–$22 per $1,000. Minimum fee: $196. A preliminary project review fee may apply for large projects. Re-inspection fees apply when a failed inspection requires another visit. Permit fees are non-refundable once the permit is issued. A refund of 80% is available if a permit is revoked before any work starts.",
                "source": "Toronto Building Fee Schedule 2024",
                "category": "Permits"
            },
            {
                "id": "bylaw_019",
                "title": "Inspections During Construction",
                "content": "Mandatory inspections must be requested by the permit holder at specific stages: 1) Excavation/foundation before concrete is poured. 2) Framing after structural framing is complete and before insulation. 3) Insulation and vapour barrier before drywall. 4) Final inspection when construction is complete. For plumbing: rough-in inspection before walls are closed. For HVAC: rough-in and final. Book inspections at least 2 business days in advance through the Toronto Building portal or 416-338-0022. Failure to call inspections can result in the City ordering work to be uncovered.",
                "source": "Ontario Building Code; Toronto Building",
                "category": "Permits"
            },
            {
                "id": "bylaw_020",
                "title": "Heritage Building Permits",
                "content": "Properties designated under the Ontario Heritage Act require a Heritage Permit in addition to a building permit for any alterations to the exterior or significant interior elements. Heritage Permit applications are reviewed by Heritage Planning staff. Changes must be consistent with the Ontario Heritage Trust Standards and Guidelines for the Conservation of Historic Places in Canada. Demolition of a heritage property is rarely permitted and requires City Council approval. Heritage Permit processing time: 60–90 days.",
                "source": "Ontario Heritage Act; Toronto Heritage Planning",
                "category": "Permits"
            },
            {
                "id": "bylaw_021",
                "title": "Electrical Safety and Electrical Permits",
                "content": "Electrical work in Ontario is regulated by the Electrical Safety Authority (ESA), not directly by the City of Toronto. An Electrical Permit is required for: new wiring, panel upgrades, installation of new outlets or circuits, installation of electric vehicle charging stations, hot tubs, and pools. Homeowners may do their own electrical work but must obtain a permit and arrange inspections. Licensed electrical contractors can also pull permits on behalf of owners. Contact ESA at 1-877-372-7233 or esasafe.com.",
                "source": "Ontario Electrical Safety Code; Electrical Safety Authority",
                "category": "Permits"
            },
            {
                "id": "bylaw_022",
                "title": "Plumbing and Drainage Permits",
                "content": "Plumbing permits are required for: new water service connections; installation or replacement of drains; installation of plumbing fixtures; backwater valve installation; disconnection or connection of drains. Backwater valve installation qualifies for a City subsidy of up to $1,250. Plumbing permits are issued through Toronto Building, not the ESA. A licensed plumber is required for most plumbing work in Toronto (some minor repairs may be done by homeowners). Permit fee: approximately $150–$350.",
                "source": "Toronto Municipal Code Chapter 363; toronto.ca/basement-flooding",
                "category": "Permits"
            },
            {
                "id": "bylaw_023",
                "title": "Demolition Permits",
                "content": "A demolition permit is required to demolish any structure. For residential buildings in urban areas, a replacement building permit must often be applied for simultaneously. Asbestos, lead paint, and other designated substances must be identified and removed by a licensed abatement contractor before demolition. The owner must notify the City at least 10 days before commencing demolition. Sewer and water connections must be properly capped. Trees on private property may be affected by the Private Tree By-law and a separate tree removal permit may be needed.",
                "source": "Toronto Municipal Code Chapter 363; Ontario Building Code Part 8",
                "category": "Permits"
            },
            {
                "id": "bylaw_024",
                "title": "Swimming Pool Enclosure and Permit",
                "content": "A building permit is required to install an in-ground or above-ground pool. All pools must be enclosed by a fence at least 1.2 m (4 feet) high. The fence must surround the pool on all four sides; a wall of the house may count as one side. Gates must be self-closing and self-latching with the latch on the inside. Pool equipment (pump, heater) must be setback from property lines per the Zoning By-law. Hot tubs follow the same enclosure rules. Failure to comply may result in a fine of up to $100,000.",
                "source": "Toronto Municipal Code Chapter 447 (Fences); Ontario Building Code",
                "category": "Permits"
            },
            {
                "id": "bylaw_025",
                "title": "Deck Permits and Requirements",
                "content": "A building permit is required for decks more than 600 mm (2 feet) above grade. Decks attached to a house must meet structural requirements (footings to frost depth). Decks must maintain minimum setbacks from property lines (same as the house setback rules in the zone). Guard rails are required when the deck surface is more than 600 mm above grade; guards must be at least 1.07 m high. Glass guards must meet impact resistance standards. Decks in rear yards under 200 m² and less than 2.5 m tall may be exempt from some setback rules.",
                "source": "Ontario Building Code; Toronto Zoning By-law 569-2013",
                "category": "Permits"
            },

            # ── WASTE MANAGEMENT ─────────────────────────────────────────────
            {
                "id": "bylaw_026",
                "title": "Blue Bin Recycling – Accepted Materials",
                "content": "Toronto's Blue Bin accepts: paper (newspapers, flyers, cardboard, boxboard, paper bags, paper cups); metal (food and beverage cans, aluminum foil, empty aerosol cans); glass (bottles and jars); plastic containers with a neck smaller than the base (bottles, jugs). NOT accepted in Blue Bin: plastic bags (return to grocery store), foam/Styrofoam, greasy pizza boxes (place in Green Bin instead), mirrors, light bulbs, syringes. Items should be empty, rinsed, and dry. Caps and lids can stay on plastic and glass bottles.",
                "source": "City of Toronto Waste Management – toronto.ca/bluebin",
                "category": "Waste"
            },
            {
                "id": "bylaw_027",
                "title": "Green Bin Organics – Accepted Materials",
                "content": "Toronto's Green Bin accepts all food waste including: meat, fish, and bones; dairy products; bread and grains; fruit and vegetables; cooking oil and fats (absorbed in paper); coffee grounds and paper filters; tea bags; soiled cardboard and paper (pizza boxes, paper napkins, paper plates). Also accepted: pet waste and pet bedding; hair and nail clippings; cut flowers; houseplants; dryer lint; biodegradable certified bags. NOT accepted: recyclables, liquids only, non-certified 'compostable' plastics.",
                "source": "City of Toronto Waste Management – toronto.ca/greenbin",
                "category": "Waste"
            },
            {
                "id": "bylaw_028",
                "title": "Garbage Collection Rules",
                "content": "Garbage (Black Bin or clear bags for apartments without bins) is collected weekly for most residences. Maximum 3 bags or equivalent per collection. Each bag must not exceed 23 kg. Garbage must be placed at the curb by 7 AM on collection day (not the night before in many areas). Bins must be removed from the curb by midnight on collection day. Items larger than 1 m must be bundled with string. Prohibited from garbage: hazardous materials, electronics, tires, batteries, propane tanks, paint.",
                "source": "City of Toronto Waste Management – toronto.ca/garbage",
                "category": "Waste"
            },
            {
                "id": "bylaw_029",
                "title": "Large Item (Bulk) Pickup",
                "content": "Large or bulky items that do not fit in bins (furniture, mattresses, appliances) can be scheduled for bulk pickup. Book online at toronto.ca/bulkpickup or call 311. Maximum 3 items per booking. Some items are free (furniture, mattresses, small appliances). Appliances containing refrigerants (fridges, AC units) are always free to pick up. Items must be placed at the curb on the scheduled date by 7 AM. Electronics (TVs, computers) are NOT included in bulk pickup; use the City's drop-off depots.",
                "source": "City of Toronto Waste Management – toronto.ca/bulkpickup",
                "category": "Waste"
            },
            {
                "id": "bylaw_030",
                "title": "Hazardous Waste Drop-Off",
                "content": "Household Hazardous Waste (HHW) must be taken to a City of Toronto drop-off depot. Accepted items: paint and stains, solvents, pesticides and herbicides, automotive fluids, propane tanks (under 20 lbs), batteries (all types), fluorescent bulbs and tubes, pool chemicals. HHW Depots are open seasonally (spring–fall) at various locations across the city. Year-round drop-off is available at some Community Environment Days. Do NOT pour HHW down the drain or place in garbage. Call 311 or visit toronto.ca/hhw for locations.",
                "source": "City of Toronto Waste Management – toronto.ca/hhw",
                "category": "Waste"
            },
            {
                "id": "bylaw_031",
                "title": "Yard Waste Collection",
                "content": "Yard waste (leaves, grass clippings, small branches) is collected separately from April to December. Bundle branches in lengths no longer than 1.2 m and no larger than 60 cm in diameter. Use paper yard waste bags or any open container. Maximum 10 bundles or bags per collection. Roots, stumps, and soil are NOT accepted. Yard waste is composted at City facilities. During the winter months (January–March), yard waste is placed in the Green Bin. Christmas trees are collected curbside in January.",
                "source": "City of Toronto Waste Management – toronto.ca/yardwaste",
                "category": "Waste"
            },
            {
                "id": "bylaw_032",
                "title": "Electronics Recycling",
                "content": "Electronics (TVs, computers, monitors, printers, phones, tablets, cables) are NOT accepted in curbside bins. Drop them off at: City of Toronto Drop-off Depots (year-round); Community Environment Days (seasonal); Retailer take-back programs (e.g., Best Buy, Staples). The Ontario Electronic Stewardship program accepts most consumer electronics free of charge. Data destruction services are available at some depots for a fee. Batteries from electronics can be removed and placed in the Blue Bin (Toronto accepts them with recycling).",
                "source": "City of Toronto Waste Management – toronto.ca/electronics",
                "category": "Waste"
            },
            {
                "id": "bylaw_033",
                "title": "Construction and Renovation Waste",
                "content": "Construction, renovation, and demolition (CRD) waste is NOT accepted in curbside bins. Options for disposal: Licensed private waste haulers (bin rental); City of Toronto Transfer Stations (fees apply; open to residents with valid permit); Habitat for Humanity ReStore (accepts salvageable building materials). Asbestos, lead, and other designated substances require licensed abatement and special disposal. Dumping CRD waste illegally is an offence punishable by a fine of up to $100,000. Contact 311 to report illegal dumping.",
                "source": "City of Toronto Waste Management; Environmental Protection Act",
                "category": "Waste"
            },

            # ── PROPERTY STANDARDS ───────────────────────────────────────────
            {
                "id": "bylaw_034",
                "title": "Exterior Property Maintenance Standards",
                "content": "Under the Property Standards By-law (Chapter 629), all property owners must maintain their property in good repair. Exterior requirements: walls, roofs, and foundations must be weathertight and free from cracking, spalling, or deterioration; windows and doors must be weather-stripped and in good working order; eavestroughs, downspouts, and drainage must be functional; exterior surfaces must be free from peeling paint, broken cladding, or graffiti; driveways and sidewalks must be maintained in a safe condition. Complaints can be filed with the City via 311.",
                "source": "Toronto Municipal Code Chapter 629 (Property Standards)",
                "category": "Property"
            },
            {
                "id": "bylaw_035",
                "title": "Interior Property Standards",
                "content": "Interior property standards apply to rental units and occupied buildings. Requirements: heating must maintain a minimum temperature of 21°C from September 15 to June 1; hot water must be maintained at a minimum of 43°C; kitchens must have a working stove and refrigerator; bathrooms must have a working toilet, sink, and either a tub or shower; electrical systems must be safe; pest-free condition must be maintained. Landlords are responsible for maintaining rental units in a good state of repair. Tenants may apply to the Landlord and Tenant Board for remedy.",
                "source": "Toronto Municipal Code Chapter 629; Residential Tenancies Act",
                "category": "Property"
            },
            {
                "id": "bylaw_036",
                "title": "Fence By-law",
                "content": "The Fence By-law (Chapter 447) regulates fence heights on residential properties. Front yard fences: maximum 1.0 m (from grade to top of fence). Rear and side yard fences: maximum 2.0 m. Fences between neighbouring residential properties are governed by the Line Fences Act (Ontario). If neighbours cannot agree on a fence, either party may apply to a fencing arbitrator. Fences made of barbed wire, razor wire, or electrified wire are prohibited in residential zones. Permits are NOT required for fences under 2.0 m in height.",
                "source": "Toronto Municipal Code Chapter 447; Line Fences Act, R.S.O. 1990",
                "category": "Property"
            },
            {
                "id": "bylaw_037",
                "title": "Grading and Drainage Standards",
                "content": "Property owners must ensure that surface drainage from their property does not flow onto neighbouring properties or cause flooding. Lots must be graded to drain toward the street or a drainage easement. Window wells must be maintained to prevent water infiltration. Sump pumps may not discharge onto the City's right-of-way without approval. The City's Basement Flooding Protection Subsidy Program offers up to $3,400 for eligible flood-protection improvements (backwater valves, sump pumps, window wells). Apply through toronto.ca/basementflooding.",
                "source": "Toronto Municipal Code Chapter 629; toronto.ca/basementflooding",
                "category": "Property"
            },
            {
                "id": "bylaw_038",
                "title": "Vacant Building Standards",
                "content": "Owners of vacant buildings must: secure all openings (doors, windows, hatchways) to prevent unauthorized entry; maintain the building's exterior to prevent deterioration; keep the property free of debris, waste, and overgrown vegetation; maintain heating if the building contains water pipes; maintain valid property insurance. The City may register a notice on title if a vacant building becomes a nuisance. Failure to comply can result in the City entering the property, performing the work, and billing the owner with the cost added to the property tax bill.",
                "source": "Toronto Municipal Code Chapter 629; City of Toronto Vacant Building Policy",
                "category": "Property"
            },
            {
                "id": "bylaw_039",
                "title": "Overgrown Vegetation and Weed Control",
                "content": "Under the Property Standards By-law, grass and weeds in non-agricultural areas must not exceed 20 cm in height. Owners or occupants must cut their grass and maintain vegetation on their property including the portion of the City boulevard between the sidewalk and the curb adjacent to their property. Hedges and shrubs must not obstruct pedestrian or vehicle sightlines. The City may cut overgrown vegetation on private property and bill the owner if the owner fails to comply after notice. Invasive species (dog-strangling vine, Japanese knotweed) must be removed.",
                "source": "Toronto Municipal Code Chapter 629; City of Toronto",
                "category": "Property"
            },

            # ── NOISE BY-LAW ─────────────────────────────────────────────────
            {
                "id": "bylaw_040",
                "title": "Construction Noise – Permitted Hours",
                "content": "The Noise By-law (Chapter 591) restricts construction noise. Permitted hours for construction noise: Monday to Friday, 7:00 AM to 7:00 PM. Saturday, 9:00 AM to 7:00 PM. Sunday and statutory holidays: construction noise is prohibited. Exemptions may be granted by the City for large infrastructure projects (TTC, road work) or emergency repairs. Persistent construction noise outside permitted hours can be reported to 311. Maximum construction noise levels may also be governed by project-specific conditions of approval.",
                "source": "Toronto Municipal Code Chapter 591 (Noise)",
                "category": "Noise"
            },
            {
                "id": "bylaw_041",
                "title": "Neighbourhood Noise – Music and Parties",
                "content": "Residential noise (music, parties, loud talking) must not unreasonably disturb neighbours at any time of day. The Noise By-law sets specific quiet hours: 11:00 PM to 7:00 AM on weekdays; 11:00 PM to 9:00 AM on weekends. Amplified music audible from neighbouring properties after these hours is a violation. Fines: $500–$5,000 depending on whether it is the first or repeated offence. Report noise complaints to Toronto Police (non-emergency) at 416-808-2222 or call 311 during business hours. Persistent issues may result in prosecution.",
                "source": "Toronto Municipal Code Chapter 591 (Noise)",
                "category": "Noise"
            },
            {
                "id": "bylaw_042",
                "title": "Commercial Noise Standards",
                "content": "Commercial operations (restaurants, bars, event venues) must not emit noise that exceeds limits set under Chapter 591 or applicable Environmental Protection Act standards. Loading docks and refrigeration units must include noise mitigation (enclosures, anti-vibration mounts). Air conditioning and HVAC equipment must comply with noise emission standards. Complaints about commercial noise can be investigated by Municipal Licensing and Standards (ML&S). Operators may be required to commission an acoustical assessment if noise complaints are substantiated.",
                "source": "Toronto Municipal Code Chapter 591; Environmental Protection Act (Ontario)",
                "category": "Noise"
            },
            {
                "id": "bylaw_043",
                "title": "Fireworks By-law",
                "content": "Display fireworks (e.g., aerial shells) require a permit from the City of Toronto Fire Services. Family/consumer fireworks (e.g., sparklers, fountains) may be used without a permit on Victoria Day and Canada Day only, between sunset and 11:00 PM. Fireworks must not be set off within 100 m of a hospital, school, or church. Discharging fireworks at other times requires written permission from the property owner and notice to neighbours. Violations are subject to fines under the Fire Prevention By-law. Contact Toronto Fire Services at 416-338-9050 for permits.",
                "source": "Toronto Municipal Code Chapter 79 (Fire Prevention); Explosives Act (Canada)",
                "category": "Noise"
            },

            # ── PARKING ──────────────────────────────────────────────────────
            {
                "id": "bylaw_044",
                "title": "On-Street Parking Rules",
                "content": "On-street parking in Toronto is governed by the Toronto Municipal Code Chapter 950 (Traffic and Parking). Generally: parking is prohibited within 9 m of an intersection; parking is prohibited on arterial roads during rush hours (7–9 AM and 4–6 PM on weekdays); maximum parking time in most residential areas is 3 hours during the day (signs indicate limits). Parking enforcement is managed by the City of Toronto. Dispute a parking ticket within 15 days at toronto.ca/parking-tickets. Tickets unpaid for 15+ days become final.",
                "source": "Toronto Municipal Code Chapter 950 (Traffic and Parking)",
                "category": "Parking"
            },
            {
                "id": "bylaw_045",
                "title": "Residential Permit Parking Program",
                "content": "The Residential Permit Parking (RPP) program allows residents to park on their street overnight (12 AM to 7 AM) and during restricted daytime hours. Annual cost: approximately $34.53 per permit (2024). Only one permit per household. Permits are street-specific and cannot be used on other streets. Apply online at toronto.ca/permits, by phone at 416-392-7873, or in person at civic centres. Eligibility requires a valid Ontario driver's licence and proof of residency on the eligible street. Permits are not transferable to a different vehicle without re-application.",
                "source": "City of Toronto Residential Permit Parking – toronto.ca/permits",
                "category": "Parking"
            },
            {
                "id": "bylaw_046",
                "title": "Accessible Parking in Toronto",
                "content": "Ontario Accessible Parking Permits (APPs) allow parking in designated accessible spaces and relief from time-restricted parking (up to 3 hours beyond posted limit). APPs are issued by ServiceOntario. Misuse of an APP (e.g., using someone else's permit) is a provincial offence with fines up to $5,000. Toronto has approximately 1,000 on-street accessible parking spaces. Designated accessible spaces must remain clear at all times. Report misuse of accessible spaces to Toronto Parking Enforcement at 416-808-2222.",
                "source": "Highway Traffic Act (Ontario); Toronto Municipal Code Chapter 950",
                "category": "Parking"
            },
            {
                "id": "bylaw_047",
                "title": "Overnight Winter Parking Ban",
                "content": "Toronto's overnight parking ban (November 15 to April 1) prohibits parking on most city streets between 12 AM and 7 AM. This ban allows snow plows to clear streets effectively. Violations result in a $100 fine and the vehicle may be towed. Residents without a driveway or garage can apply for a Residential Permit Parking permit (see bylaw_045). The overnight ban does not apply on streets with no-stopping signs or where daytime parking is also restricted. Check toronto.ca/parking for specific street information.",
                "source": "Toronto Municipal Code Chapter 950; City of Toronto Winter Parking",
                "category": "Parking"
            },
            {
                "id": "bylaw_048",
                "title": "Recreational Vehicle and Boat Storage",
                "content": "Trailers, boats, and recreational vehicles (RVs) parked on a City street are subject to the 3-hour daytime parking limit and overnight parking ban. On private property, an RV or trailer may be stored in a side or rear yard only (not the front yard) in residential zones. RVs and trailers must not be used for habitation while parked on a residential property unless permitted by a valid campground licence. Commercial vehicles over 3 tonnes may not be parked on residential streets overnight.",
                "source": "Toronto Municipal Code Chapter 950; Toronto Zoning By-law 569-2013",
                "category": "Parking"
            },

            # ── PARKS AND RECREATION ─────────────────────────────────────────
            {
                "id": "bylaw_049",
                "title": "Park Use Rules and Hours",
                "content": "Toronto parks are generally open from 5:30 AM to midnight daily, unless posted signs indicate otherwise. Alcohol is prohibited in all parks unless a permit has been issued for a special event. Motorized vehicles are prohibited on park paths and grass. Camping overnight in parks (other than designated camping areas) is prohibited. Open fires are prohibited in parks except in designated fire pits. Picnicking is permitted in all parks. Groups of more than 25 people may require a park use permit for organized activities. Apply through toronto.ca/permits.",
                "source": "Toronto Municipal Code Chapter 608 (Parks)",
                "category": "Parks"
            },
            {
                "id": "bylaw_050",
                "title": "Dogs in Parks – Off-Leash Areas",
                "content": "Dogs must be on a leash in all Toronto parks except in designated off-leash areas (OLAs). There are over 60 designated off-leash areas across the city. Dogs in OLAs must still be under voice control and owners must clean up after their dogs. In non-OLA areas, the fine for an off-leash dog is $365. Dogs are prohibited from beaches, wading pools, sports fields, and playgrounds at all times. To find your nearest OLA, visit toronto.ca/offleash. All dogs over 6 months in Toronto must be licensed annually (fee: $50 for spayed/neutered dogs, $90 for intact dogs).",
                "source": "Toronto Municipal Code Chapter 608; Municipal Code Chapter 349 (Animals)",
                "category": "Parks"
            },
            {
                "id": "bylaw_051",
                "title": "Special Events in Parks",
                "content": "Events in Toronto parks with amplified sound, food sales, alcohol, or more than 25 attendees require a Park Use Permit. Applications must be submitted at least 4 weeks in advance (larger events: 16 weeks). Fees depend on event size and location (starting at approximately $100). Additional approvals may be needed from Toronto Fire Services, Municipal Licensing & Standards (if selling food), and Toronto Public Health. Event organizers are responsible for cleanup and must leave the park in its original condition. Apply online at toronto.ca/permits.",
                "source": "Toronto Municipal Code Chapter 608 (Parks); Special Events Office",
                "category": "Parks"
            },

            # ── TREE PROTECTION ──────────────────────────────────────────────
            {
                "id": "bylaw_052",
                "title": "Private Tree Protection By-law",
                "content": "The City of Toronto's Private Tree By-law (Municipal Code Chapter 813) protects trees on private property with a trunk diameter of 30 cm or more (measured at 1.4 m above grade). A permit is required to injure or remove a protected private tree. The permit application requires a certified arborist's report. The City may require tree preservation measures, protective fencing, and replanting as conditions of approval. Unauthorized removal or injury of a protected tree can result in fines of $500–$100,000 per tree or order to replant.",
                "source": "Toronto Municipal Code Chapter 813 (Trees); toronto.ca/trees",
                "category": "Trees"
            },
            {
                "id": "bylaw_053",
                "title": "Street Tree Protection",
                "content": "City-owned street trees (trees in the City's right-of-way) may not be damaged, removed, or built around without written approval from Urban Forestry. Approval is required before any excavation within the dripline of a street tree. Protective fencing (orange snow fence) is mandatory during any construction near street trees. Damaging a street tree is a violation punishable by fines comparable to private tree violations. The City plants approximately 100,000 trees annually through the City's tree planting program. To request a new street tree, contact 311.",
                "source": "Toronto Municipal Code Chapter 813; Urban Forestry – toronto.ca/urbanforestry",
                "category": "Trees"
            },
            {
                "id": "bylaw_054",
                "title": "Tree Removal Permit Process",
                "content": "To remove a protected private tree (≥30 cm diameter at breast height), file a tree removal application at toronto.ca/trees or in person at a Civic Centre. Include: completed application form; certified arborist's report with species, size, condition, and reason for removal; site plan showing tree location and proposed construction. Processing time: 30–60 business days. Trees in poor condition (arborist-certified) are usually approved for removal. Healthy trees may require replacement planting. Replacement ratio is typically 1:1 (one replacement tree per tree removed).",
                "source": "Toronto Municipal Code Chapter 813; toronto.ca/trees",
                "category": "Trees"
            },
            {
                "id": "bylaw_055",
                "title": "Ravine and Natural Feature Protection",
                "content": "The Ravine and Natural Feature Protection By-law (By-law 2-2017) prohibits injury to trees and vegetation, grading, filling, or dumping in Toronto's ravines and natural heritage areas. A Ravine Special Permit is required for any work within 10 m of the crest of a ravine slope. Activities permitted without a permit include mowing grass, removing invasive species, and walking. The Toronto and Region Conservation Authority (TRCA) also regulates development near regulated areas (floodplains, wetlands). TRCA approval is separate from the City's ravine permit. Contact TRCA at trca.ca.",
                "source": "City of Toronto Ravine and Natural Feature Protection By-law 2-2017; TRCA",
                "category": "Trees"
            },

            # ── ADDITIONAL / MISCELLANEOUS ───────────────────────────────────
            {
                "id": "bylaw_056",
                "title": "Business Licensing in Toronto",
                "content": "Many business types require a Municipal Licence from the City of Toronto. Licensed businesses include: restaurants and food establishments, body rub parlours, tow truck operators, vehicle-for-hire (taxis, Uber, Lyft), rooming houses, pet shops, and amusement arcades. Apply through Municipal Licensing and Standards (ML&S) at toronto.ca/business. Requirements typically include: application form, applicable fees, proof of insurance, zoning compliance, and in some cases, inspections. Operating without a required licence is an offence subject to fines.",
                "source": "Toronto Municipal Code Chapter 545 (Licensing); toronto.ca/business",
                "category": "Business"
            },
            {
                "id": "bylaw_057",
                "title": "Food Premises Standards",
                "content": "Food establishments must be licensed by ML&S and inspected by Toronto Public Health (TPH). Inspections check: food storage temperatures (hot foods ≥60°C, cold foods ≤4°C); personal hygiene of food handlers; pest control; cleanliness of equipment and premises; food source documentation. Inspection results (Pass, Conditional Pass, Closed) are posted on the DineSafe program at toronto.ca/dinesafe. Critical infractions require immediate correction. Operators must complete a Food Handler Certification course. Report food safety concerns to toronto.ca/foodsafety or 311.",
                "source": "Toronto Municipal Code Chapter 545; Health Protection and Promotion Act (Ontario)",
                "category": "Business"
            },
            {
                "id": "bylaw_058",
                "title": "Home Occupation (Working from Home) Rules",
                "content": "Operating a business from a residential property (home occupation) is permitted in most Toronto residential zones, subject to conditions: the occupation must be secondary to the residential use; the business must be conducted entirely within the dwelling (no outdoor storage); no more than one non-resident employee on the premises at a time; no signage visible from the street; no retail sales to the public; no noise, odour, or traffic in excess of residential norms. Permitted home occupations include offices, studios, tutoring, and professional services. Not permitted: auto repair, manufacturing, food processing.",
                "source": "Toronto Zoning By-law 569-2013, Section 150",
                "category": "Business"
            },
            {
                "id": "bylaw_059",
                "title": "Sign By-law – General Provisions",
                "content": "Toronto's Sign By-law (Chapter 693) regulates the number, size, type, and placement of signs on all properties. Most signs require a sign permit from the City. Exceptions: window signs up to 20% of window area; small real estate signs on private property; temporary sale signs. Prohibited signs: roof signs, signs illuminated with strobe lights, signs in park or ravine areas. Third-party advertising signs (billboards) require a Third-party Sign Permit and may only be located in certain commercial and employment zones. Apply for a sign permit at toronto.ca/signs.",
                "source": "Toronto Municipal Code Chapter 693 (Signs)",
                "category": "Signs"
            },
            {
                "id": "bylaw_060",
                "title": "Election Signs",
                "content": "Election signs may be placed on private property with owner consent during election periods. On City property (boulevards, utility poles), election signs are permitted only between the date the writ of election is issued and 24 hours after polls close. Signs must not obstruct sightlines at intersections. Signs must display the name of the candidate or elector organization. Maximum size for election signs on private property: 1.5 m × 1.5 m. Removal of signs on public property after the election period by City staff may result in the candidate being billed for removal costs.",
                "source": "Toronto Municipal Code Chapter 693; City of Toronto Election Sign Policy",
                "category": "Signs"
            },
            {
                "id": "bylaw_061",
                "title": "Toronto 311 – Service Request and By-law Complaints",
                "content": "Toronto 311 is the City's primary non-emergency contact for all municipal service requests and by-law complaints. Available: 24 hours a day, 7 days a week, 365 days a year by phone (311) or online at toronto.ca/311. Services accessible through 311: report potholes, sidewalk damage, fallen trees, graffiti, illegal dumping, noise complaints, property standards complaints, and parking violations. 311 agents can create service requests, provide information, and direct callers to the right City division. Service request status can be tracked online with the confirmation number provided.",
                "source": "City of Toronto – toronto.ca/311",
                "category": "General"
            },
            {
                "id": "bylaw_062",
                "title": "Municipal Accommodation Tax (Hotels and Short-Term Rentals)",
                "content": "Toronto's Municipal Accommodation Tax (MAT) is 6% of the price of accommodation for stays of less than 28 consecutive nights. Applies to: hotels, motels, bed and breakfasts, short-term rental platforms (Airbnb, VRBO). Operators must collect the tax from guests and remit it to the City quarterly. Exemption: stays 29 days or more are not subject to MAT. Revenue from the MAT is split: 50% to Tourism Toronto and 50% to the City's general revenue. Register and remit at toronto.ca/mat.",
                "source": "Toronto Municipal Code Chapter 758 (Municipal Accommodation Tax)",
                "category": "Business"
            },
            {
                "id": "bylaw_063",
                "title": "Stormwater Management and Downspout Disconnection",
                "content": "Toronto requires downspout disconnection from the sanitary/combined sewer to reduce basement flooding. Downspouts must discharge onto a splash pad and direct water to a vegetated area, rain barrel, or the City's right-of-way. The Downspout Disconnection Program provides free inspection and disconnection for eligible homes. The City also offers a Stormwater Charge on all water bills to fund infrastructure upgrades. Rain garden and green roof grants are available to help manage stormwater on private property. Contact 311 or visit toronto.ca/stormwater for details.",
                "source": "City of Toronto Stormwater Management; toronto.ca/stormwater",
                "category": "Property"
            },
            {
                "id": "bylaw_064",
                "title": "Graffiti By-law",
                "content": "The Graffiti By-law (Chapter 485) prohibits graffiti on any public or private surface without authorization. Property owners must remove graffiti from their property within 30 days of it appearing, or within 5 business days if notified by the City. The City's Graffiti Management Program provides free graffiti removal on private property facing the public right-of-way (first time only). Report graffiti at toronto.ca/graffiti or 311. Tags on utility boxes may be reported to the utility (e.g., Hydro One, Toronto Hydro). A $360 fine applies for failing to remove graffiti after notice.",
                "source": "Toronto Municipal Code Chapter 485 (Graffiti); toronto.ca/graffiti",
                "category": "Property"
            },
            {
                "id": "bylaw_065",
                "title": "Animal Licensing and Dog Licensing",
                "content": "All dogs kept in Toronto must be licensed annually. Licence fees: $50 for spayed/neutered dogs; $90 for intact (unaltered) dogs. Licences are sold at Pet Smart, City of Toronto Animal Services, and online at toronto.ca/animallicences. Lost dogs with a valid City licence can be reunited with their owner more easily. Cats must be microchipped and vaccinated for rabies; cat licensing is encouraged but not mandatory. Keeping more than 3 dogs in a residential unit requires a kennel licence. Toronto Animal Services can be reached at 416-338-8723.",
                "source": "Toronto Municipal Code Chapter 349 (Animals); toronto.ca/animallicences",
                "category": "General"
            }
        ]

    def initialize_knowledge_base(self):
        """Initialize ChromaDB with knowledge base documents."""
        if self.collection.count() > 0:
            print("[RAG] Knowledge base already initialized")
            return

        for doc in self.knowledge_base:
            self.collection.add(
                ids=[doc["id"]],
                documents=[doc["content"]],
                metadatas=[{
                    "title": doc["title"],
                    "source": doc["source"],
                    "category": doc.get("category", "General")
                }]
            )
        print(f"[RAG] Initialized knowledge base with {len(self.knowledge_base)} documents")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        documents = []
        for i, doc in enumerate(results["documents"][0]):
            documents.append({
                "title": results["metadatas"][0][i].get("title", ""),
                "content": doc,
                "source": results["metadatas"][0][i].get("source", ""),
                "category": results["metadatas"][0][i].get("category", "")
            })

        return documents


# Global RAG instance
rag = TorontoBylawRAG()
