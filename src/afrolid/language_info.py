from enum import Enum
from typing import Final, TypedDict

class Language(Enum):
    MDA = {'name': 'Mada', 'script': 'Latin'}
    NIN = {'name': 'Ninzo', 'script': 'Latin'}
    ODU = {'name': 'Odual', 'script': 'Latin'}
    ABN = {'name': 'Abua', 'script': 'Latin'}
    EGO = {'name': 'Eggon', 'script': 'Latin'}
    KYQ = {'name': 'Kenga', 'script': 'Latin'}
    BDH = {'name': 'Baka', 'script': 'Latin'}
    EKA = {'name': 'Ekajuk', 'script': 'Latin'}
    BZA = {'name': 'Bandi', 'script': 'Latin'}
    BFA = {'name': 'Bari', 'script': 'Latin'}
    AHA = {'name': 'Ahanta', 'script': 'Latin'}
    BOX = {'name': 'Bwamu / Buamu', 'script': 'Latin'}
    WBI = {'name': 'Vwanji', 'script': 'Latin'}
    MWE = {'name': 'Mwera', 'script': 'Latin'}
    ASA = {'name': 'Asu', 'script': 'Latin'}
    BEM = {'name': 'Bemba / Chibemba', 'script': 'Latin'}
    BEQ = {'name': 'Beembe', 'script': 'Latin'}
    BEZ = {'name': 'Bena', 'script': 'Latin'}
    BXK = {'name': 'Bukusu', 'script': 'Latin'}
    CCE = {'name': 'Chopi', 'script': 'Latin'}
    CHW = {'name': 'Chuabo', 'script': 'Latin'}
    CJK = {'name': 'Chokwe', 'script': 'Latin'}
    CWE = {'name': 'Kwere', 'script': 'Latin'}
    DAV = {'name': 'Dawida / Taita', 'script': 'Latin'}
    DHM = {'name': 'Dhimba', 'script': 'Latin'}
    DIG = {'name': 'Chidigo', 'script': 'Latin'}
    DIU = {'name': 'Gciriku', 'script': 'Latin'}
    DUG = {'name': 'Chiduruma', 'script': 'Latin'}
    EBU = {'name': 'Kiembu / Embu', 'script': 'Latin'}
    EKO = {'name': 'Koti', 'script': 'Latin'}
    FIP = {'name': 'Fipa', 'script': 'Latin'}
    FLR = {'name': 'Fuliiru', 'script': 'Latin'}
    GOG = {'name': 'Gogo', 'script': 'Latin'}
    GUZ = {'name': 'Ekegusii', 'script': 'Latin'}
    GWR = {'name': 'Gwere', 'script': 'Latin'}
    HAY = {'name': 'Haya', 'script': 'Latin'}
    JIT = {'name': 'Jita', 'script': 'Latin'}
    JMC = {'name': 'Machame', 'script': 'Latin'}
    KAM = {'name': 'Kikamba', 'script': 'Latin'}
    KCK = {'name': 'Kalanga', 'script': 'Latin'}
    KDC = {'name': 'Kutu', 'script': 'Latin'}
    KDE = {'name': 'Makonde', 'script': 'Latin'}
    KDN = {'name': 'Kunda', 'script': 'Latin'}
    KIK = {'name': 'Gikuyu / Kikuyu', 'script': 'Latin'}
    KIN = {'name': 'Kinyarwanda', 'script': 'Latin'}
    KIZ = {'name': 'Kisi', 'script': 'Latin'}
    KKI = {'name': 'Kagulu', 'script': 'Latin'}
    KMB = {'name': 'Kimbundu', 'script': 'Latin'}
    KNG = {'name': 'Kongo', 'script': 'Latin'}
    KOO = {'name': 'Konzo', 'script': 'Latin'}
    KQN = {'name': 'Kikaonde', 'script': 'Latin'}
    KSB = {'name': 'Shambala / Kishambala', 'script': 'Latin'}
    KUA = {'name': 'Oshiwambo', 'script': 'Latin'}
    KUJ = {'name': 'Kuria', 'script': 'Latin'}
    KWN = {'name': 'Kwangali', 'script': 'Latin'}
    LAI = {'name': 'Lambya', 'script': 'Latin'}
    LAM = {'name': 'Lamba', 'script': 'Latin'}
    LGM = {'name': 'Lega-mwenga', 'script': 'Latin'}
    LIK = {'name': 'Lika', 'script': 'Latin'}
    MCK = {'name': 'Mbunda', 'script': 'Latin'}
    MER = {'name': 'Kimiiru', 'script': 'Latin'}
    MGH = {'name': 'Makhuwa-Meetto', 'script': 'Latin'}
    MGQ = {'name': 'Malila', 'script': 'Latin'}
    MGR = {'name': 'Mambwe-Lungu', 'script': 'Latin'}
    MGW = {'name': 'Matumbi', 'script': 'Latin'}
    MWS = {'name': 'Mwimbi-Muthambi', 'script': 'Latin'}
    MYX = {'name': 'Masaaba', 'script': 'Latin'}
    NBA = {'name': 'Nyemba', 'script': 'Latin'}
    NBL = {'name': 'IsiNdebele', 'script': 'Latin'}
    NDC = {'name': 'Ndau', 'script': 'Latin'}
    NDE = {'name': 'IsiNdebele', 'script': 'Latin'}
    NDH = {'name': 'Ndali', 'script': 'Latin'}
    NDJ = {'name': 'Ndamba', 'script': 'Latin'}
    NDO = {'name': 'Ndonga', 'script': 'Latin'}
    NGL = {'name': 'Lomwe', 'script': 'Latin'}
    NGO = {'name': 'Ngoni', 'script': 'Latin'}
    NGP = {'name': 'Ngulu', 'script': 'Latin'}
    NIH = {'name': 'Nyiha', 'script': 'Latin'}
    NIM = {'name': 'Nilamba / kinilyamba', 'script': 'Latin'}
    NKA = {'name': 'Nkoya / ShiNkoya', 'script': 'Latin'}
    NNQ = {'name': 'Ngindo', 'script': 'Latin'}
    NSE = {'name': 'Chinsenga', 'script': 'Latin'}
    NSO = {'name': 'Sepedi', 'script': 'Latin'}
    NUJ = {'name': 'Nyole', 'script': 'Latin'}
    NYA = {'name': 'Chichewa', 'script': 'Latin'}
    NYD = {'name': 'Olunyole / Nyore', 'script': 'Latin'}
    NYF = {'name': 'Giryama', 'script': 'Latin'}
    NYK = {'name': 'Nyaneka', 'script': 'Latin'}
    NYM = {'name': 'Nyamwezi', 'script': 'Latin'}
    NYN = {'name': 'Nyankore / Nyankole', 'script': 'Latin'}
    NYO = {'name': 'Nyoro', 'script': 'Latin'}
    NYU = {'name': 'Nyungwe', 'script': 'Latin'}
    NYY = {'name': 'Nyakyusa-Ngonde / Kyangonde', 'script': 'Latin'}
    POY = {'name': 'Pogolo / Shipogoro-Pogolo', 'script': 'Latin'}
    RAG = {'name': 'Lulogooli', 'script': 'Latin'}
    RIM = {'name': 'Nyaturu', 'script': 'Latin'}
    RND = {'name': 'Uruund', 'script': 'Latin'}
    RNG = {'name': 'Ronga / ShiRonga', 'script': 'Latin'}
    RUB = {'name': 'Gungu', 'script': 'Latin'}
    RUN = {'name': 'Rundi / Kirundi', 'script': 'Latin'}
    RWK = {'name': 'Rwa', 'script': 'Latin'}
    SBP = {'name': 'Sangu', 'script': 'Latin'}
    SBS = {'name': 'Kuhane', 'script': 'Latin'}
    SBY = {'name': 'Soli', 'script': 'Latin'}
    SNA = {'name': 'Shona', 'script': 'Latin'}
    SNG = {'name': 'Sanga / Kiluba', 'script': 'Latin'}
    SOP = {'name': 'Kisonge', 'script': 'Latin'}
    SOT = {'name': 'Sesotho', 'script': 'Latin'}
    SSW = {'name': 'Siswati', 'script': 'Latin'}
    SUK = {'name': 'Sukuma', 'script': 'Latin'}
    SWA = {'name': 'Swahili', 'script': 'Latin'}
    SWC = {'name': 'Swahili Congo', 'script': 'Latin'}
    SWH = {'name': 'Swahili', 'script': 'Latin'}
    SWK = {'name': 'Sena, Malawi', 'script': 'Latin'}
    SXB = {'name': 'Suba', 'script': 'Latin'}
    THK = {'name': 'Tharaka', 'script': 'Latin'}
    TKE = {'name': 'Takwane', 'script': 'Latin'}
    TLJ = {'name': 'Talinga-Bwisi', 'script': 'Latin'}
    TOG = {'name': 'Tonga', 'script': 'Latin'}
    TOH = {'name': 'Gitonga', 'script': 'Latin'}
    TOI = {'name': 'Chitonga', 'script': 'Latin'}
    TSC = {'name': 'Tshwa', 'script': 'Latin'}
    TSN = {'name': 'Setswana', 'script': 'Latin'}
    TSO = {'name': 'Tsonga', 'script': 'Latin'}
    TTJ = {'name': 'Toro / Rutoro', 'script': 'Latin'}
    TUM = {'name': 'Chitumbuka', 'script': 'Latin'}
    UMB = {'name': 'Umbundu', 'script': 'Latin'}
    VEN = {'name': 'Tshivenda', 'script': 'Latin'}
    VID = {'name': 'Chividunda', 'script': 'Latin'}
    VIF = {'name': 'Vili', 'script': 'Latin'}
    VMK = {'name': 'Makhuwa-Shirima', 'script': 'Latin'}
    VMW = {'name': 'Macua', 'script': 'Latin'}
    VUN = {'name': 'Kivunjo', 'script': 'Latin'}
    WMW = {'name': 'Mwani', 'script': 'Latin'}
    XHO = {'name': 'Isixhosa', 'script': 'Latin'}
    XOG = {'name': 'Soga', 'script': 'Latin'}
    YAO = {'name': 'Yao / Chiyao', 'script': 'Latin'}
    YOM = {'name': 'Ibinda', 'script': 'Latin'}
    ZAJ = {'name': 'Zaramo', 'script': 'Latin'}
    ZDJ = {'name': 'Comorian, Ngazidja', 'script': 'Latin'}
    ZGA = {'name': 'Kinga', 'script': 'Latin'}
    ZIW = {'name': 'Zigula', 'script': 'Latin'}
    ZUL = {'name': 'Isizulu', 'script': 'Latin'}
    MWN = {'name': 'Cinamwanga', 'script': 'Latin'}
    LOZ = {'name': 'Silozi', 'script': 'Latin'}
    LSM = {'name': 'Saamya-Gwe / Saamia', 'script': 'Latin'}
    LTO = {'name': 'Tsotso', 'script': 'Latin'}
    LUA = {'name': 'Tshiluba', 'script': 'Latin'}
    LUE = {'name': 'Luvale', 'script': 'Latin'}
    LUG = {'name': 'Luganda', 'script': 'Latin'}
    LUN = {'name': 'Lunda', 'script': 'Latin'}
    LWG = {'name': 'Wanga', 'script': 'Latin'}
    DWR = {'name': 'Dawro', 'script': 'Latin'}
    GMV = {'name': 'Gamo', 'script': 'Latin'}
    GOF = {'name': 'Goofa', 'script': 'Latin'}
    WAL = {'name': 'Wolaytta', 'script': 'Latin'}
    BCI = {'name': 'Baoulé', 'script': 'Latin'}
    CKO = {'name': 'Anufo', 'script': 'Latin'}
    FAT = {'name': 'Fante', 'script': 'Latin'}
    NZI = {'name': 'Nzema', 'script': 'Latin'}
    SFW = {'name': 'Sehwi', 'script': 'Latin'}
    TWI = {'name': 'Twi', 'script': 'Latin'}
    HEH = {'name': 'Hehe', 'script': 'Latin'}
    HER = {'name': 'Herero', 'script': 'Latin'}
    PEM = {'name': 'Kipende', 'script': 'Latin'}
    PKB = {'name': 'Kipfokomo / Pokomo', 'script': 'Latin'}
    DIB = {'name': 'Dinka, South Central', 'script': 'Latin'}
    DIK = {'name': 'Dinka, Southwestern', 'script': 'Latin'}
    DIP = {'name': 'Dinka, Northeastern', 'script': 'Latin'}
    DKS = {'name': 'Dinka, Southeastern', 'script': 'Latin'}
    NUS = {'name': 'Nuer', 'script': 'Latin'}
    DOW = {'name': 'Doyayo', 'script': 'Latin'}
    KMY = {'name': 'Koma', 'script': 'Latin'}
    GKN = {'name': 'Gokana', 'script': 'Latin'}
    OGO = {'name': 'Khana', 'script': 'Latin'}
    KQY = {'name': 'Koorete', 'script': 'Latin'}
    FUH = {'name': 'Fulfulde, Western Niger', 'script': 'Latin'}
    FUQ = {'name': 'Fulfulde Central Eastern Niger', 'script': 'Latin'}
    FUV = {'name': 'Fulfulde Nigeria', 'script': 'Arabic'}
    FUB = {'name': 'Fulfulde, Adamawa', 'script': 'Latin'}
    GNA = {'name': 'Kaansa', 'script': 'Latin'}
    KBP = {'name': 'Kabiye', 'script': 'Latin'}
    KDH = {'name': 'Tem', 'script': 'Latin'}
    LEE = {'name': 'Lyélé', 'script': 'Latin'}
    MZW = {'name': 'Deg', 'script': 'Latin'}
    NNW = {'name': 'Nuni, Southern', 'script': 'Latin'}
    NTR = {'name': 'Delo', 'script': 'Latin'}
    SIG = {'name': 'Paasaal', 'script': 'Latin'}
    SIL = {'name': 'Sisaala, Tumulung', 'script': 'Latin'}
    TPM = {'name': 'Tampulma', 'script': 'Latin'}
    XSM = {'name': 'Kasem', 'script': 'Latin'}
    VAG = {'name': 'Vagla', 'script': 'Latin'}
    ACD = {'name': 'Gikyode', 'script': 'Latin'}
    NAW = {'name': 'Nawuri', 'script': 'Latin'}
    NCU = {'name': 'Chunburung', 'script': 'Latin'}
    NKO = {'name': 'Nkonya', 'script': 'Latin'}
    IDU = {'name': 'Idoma', 'script': 'Latin'}
    IGE = {'name': 'Igede', 'script': 'Latin'}
    YBA = {'name': 'Yala', 'script': 'Latin'}
    BQJ = {'name': 'Bandial', 'script': 'Latin'}
    CSK = {'name': 'Jola Kasa', 'script': 'Latin'}
    JIB = {'name': 'Jibu', 'script': 'Latin'}
    NNB = {'name': 'Nande / Ndandi', 'script': 'Latin'}
    KLN = {'name': 'Kalenjin', 'script': 'Latin'}
    PKO = {'name': 'Pökoot', 'script': 'Latin'}
    KRX = {'name': 'Karon', 'script': 'Latin'}
    KIA = {'name': 'Kim', 'script': 'Latin'}
    CME = {'name': 'Cerma', 'script': 'Latin'}
    AKP = {'name': 'Siwu', 'script': 'Latin'}
    LEF = {'name': 'Lelemi', 'script': 'Latin'}
    LIP = {'name': 'Sekpele', 'script': 'Latin'}
    SNW = {'name': 'Selee', 'script': 'Latin'}
    KDJ = {'name': 'Ng’akarimojong', 'script': 'Latin'}
    LOT = {'name': 'Latuka', 'script': 'Latin'}
    MAS = {'name': 'Maasai', 'script': 'Latin'}
    SAQ = {'name': 'Samburu', 'script': 'Latin'}
    TEO = {'name': 'Teso', 'script': 'Latin'}
    TUV = {'name': 'Turkana', 'script': 'Latin'}
    ADH = {'name': 'Jopadhola / Adhola', 'script': 'Latin'}
    ALZ = {'name': 'Alur', 'script': 'Latin'}
    ANU = {'name': 'Anyuak / Anuak', 'script': 'Latin'}
    KDI = {'name': 'Kumam', 'script': 'Latin'}
    LAJ = {'name': 'Lango', 'script': 'Latin'}
    LTH = {'name': 'Thur / Acholi-Labwor', 'script': 'Latin'}
    LUO = {'name': 'Dholuo / Luo', 'script': 'Latin'} 
    LWO = {'name': 'Luwo', 'script': 'Latin'} 
    MFZ = {'name': 'Mabaan', 'script': 'Latin'} 
    SHK = {'name': 'Shilluk', 'script': 'Latin'} 
    ACH = {'name': 'Acholi', 'script': 'Latin'} 
    MPE = {'name': 'Majang', 'script': 'Latin'} 
    MCU = {'name': 'Mambila, Cameroon', 'script': 'Latin'} 
    BAM = {'name': 'Bambara', 'script': 'Latin'} 
    DYU = {'name': 'Jula', 'script': 'Latin'} 
    KNK = {'name': 'Kuranko', 'script': 'Latin'} 
    MSC = {'name': 'Maninka, Sankaran', 'script': 'Latin'} 
    MFG = {'name': 'Mogofin', 'script': 'Latin'} 
    MNK = {'name': 'Mandinka', 'script': 'Latin'} 
    NZA = {'name': 'Mbembe, Tigon', 'script': 'Latin'} 
    KBN = {'name': 'Kare', 'script': 'Latin'} 
    KZR = {'name': 'Karang', 'script': 'Latin'} 
    TUI = {'name': 'Toupouri', 'script': 'Latin'} 
    XUO = {'name': 'Kuo', 'script': 'Latin'} 
    MEN = {'name': 'Mende', 'script': 'Latin'} 
    BEX = {'name': 'Jur Modo', 'script': 'Latin'} 
    MGC = {'name': 'Morokodo', 'script': 'Latin'} 
    BCN = {'name': 'Bali', 'script': 'Latin'} 
    MZM = {'name': 'Mumuye', 'script': 'Latin'} 
    AGQ = {'name': 'Aghem', 'script': 'Latin'} 
    AZO = {'name': 'Awing', 'script': 'Latin'} 
    BAV = {'name': 'Vengo', 'script': 'Latin'} 
    BBJ = {'name': "Ghomálá'", 'script': 'Latin'} 
    BBK = {'name': 'Babanki', 'script': 'Latin'} 
    BFD = {'name': 'Bafut', 'script': 'Latin'} 
    BMO = {'name': 'Bambalang', 'script': 'Latin'} 
    BMV = {'name': 'Bum', 'script': 'Latin'} 
    BYV = {'name': 'Medumba', 'script': 'Latin'} 
    JGO = {'name': 'Ngomba', 'script': 'Latin'} 
    LMP = {'name': 'Limbum', 'script': 'Latin'} 
    MGO = {'name': "Meta'", 'script': 'Latin'} 
    MNF = {'name': 'Mundani', 'script': 'Latin'} 
    NGN = {'name': 'Bassa', 'script': 'Latin'} 
    NLA = {'name': 'Ngombale', 'script': 'Latin'} 
    NNH = {'name': 'Ngiemboon', 'script': 'Latin'} 
    OKU = {'name': 'Oku', 'script': 'Latin'} 
    YAM = {'name': 'Yamba', 'script': 'Latin'} 
    YBB = {'name': 'Yemba', 'script': 'Latin'} 
    MDM = {'name': 'Mayogo', 'script': 'Latin'} 
    MBU = {'name': 'Mbula-Bwazza', 'script': 'Latin'} 
    GJN = {'name': 'Gonja', 'script': 'Latin'} 
    BUY = {'name': 'Bullom So', 'script': 'Latin'} 
    GYA = {'name': 'Gbaya, Northwest', 'script': 'Latin'} 
    BSS = {'name': 'Akoose', 'script': 'Latin'} 
    BUM = {'name': 'Bulu', 'script': 'Latin'} 
    DUA = {'name': 'Douala', 'script': 'Latin'} 
    ETO = {'name': 'Eton', 'script': 'Latin'} 
    EWO = {'name': 'Ewondo', 'script': 'Latin'} 
    IYX = {'name': 'yaka', 'script': 'Latin'} 
    KHY = {'name': 'Kele / Lokele', 'script': 'Latin'} 
    KKJ = {'name': 'Kako', 'script': 'Latin'} 
    KOQ = {'name': 'Kota', 'script': 'Latin'} 
    KSF = {'name': 'Bafia', 'script': 'Latin'} 
    LIN = {'name': 'Lingala', 'script': 'Latin'} 
    MCP = {'name': 'Makaa', 'script': 'Latin'} 
    NGC = {'name': 'Ngombe', 'script': 'Latin'} 
    NXD = {'name': 'Ngando', 'script': 'Latin'} 
    OZM = {'name': 'Koonzime', 'script': 'Latin'} 
    TLL = {'name': 'Otetela', 'script': 'Latin'} 
    TVU = {'name': 'Tunen', 'script': 'Latin'} 
    WON = {'name': 'Wongo', 'script': 'Latin'} 
    YAT = {'name': 'Yambeta', 'script': 'Latin'} 
    LEM = {'name': 'Nomaande', 'script': 'Latin'} 
    LOQ = {'name': 'Lobala', 'script': 'Latin'} 
    IBB = {'name': 'Ibibio', 'script': 'Latin'} 
    EFI = {'name': 'Efik', 'script': 'Latin'} 
    ANN = {'name': 'Obolo', 'script': 'Latin'} 
    BFO = {'name': 'Birifor, Malba', 'script': 'Latin'} 
    BIM = {'name': 'Bimoba', 'script': 'Latin'} 
    BIV = {'name': 'Birifor, Southern', 'script': 'Latin'} 
    BUD = {'name': 'Ntcham', 'script': 'Latin'} 
    BWU = {'name': 'Buli', 'script': 'Latin'} 
    DAG = {'name': 'Dagbani', 'script': 'Latin'} 
    DGA = {'name': 'Dagaare', 'script': 'Latin'} 
    DGD = {'name': 'Dagaari Dioula', 'script': 'Latin'} 
    DGI = {'name': 'Dagara, Northern', 'script': 'Latin'} 
    GNG = {'name': 'Ngangam', 'script': 'Latin'} 
    GUR = {'name': 'Farefare', 'script': 'Latin'} 
    GUX = {'name': 'Gourmanchema', 'script': 'Latin'} 
    HAG = {'name': 'Hanga', 'script': 'Latin'} 
    KMA = {'name': 'Konni', 'script': 'Latin'} 
    KUS = {'name': 'Kusaal', 'script': 'Latin'} 
    MAW = {'name': 'Mampruli', 'script': 'Latin'} 
    MFQ = {'name': 'Moba', 'script': 'Latin'} 
    MOS = {'name': 'Moore', 'script': 'Latin'} 
    SOY = {'name': 'Miyobe', 'script': 'Latin'} 
    XON = {'name': 'Konkomba', 'script': 'Latin'} 
    MWM = {'name': 'Sar', 'script': 'Latin'} 
    MYB = {'name': 'Mbay', 'script': 'Latin'} 
    BJV = {'name': 'Bedjond', 'script': 'Latin'} 
    GQR = {'name': 'Gor', 'script': 'Latin'} 
    GVL = {'name': 'Gulay', 'script': 'Latin'} 
    KSP = {'name': 'Kabba', 'script': 'Latin'} 
    LAP = {'name': 'Laka', 'script': 'Latin'} 
    SBA = {'name': 'Ngambay', 'script': 'Latin'} 
    NDZ = {'name': 'Ndogo', 'script': 'Latin'} 
    LNL = {'name': 'Banda, South Central', 'script': 'Latin'} 
    BUN = {'name': 'Sherbro', 'script': 'Latin'} 
    GSO = {'name': 'Gbaya, Southwest', 'script': 'Latin'} 
    MUR = {'name': 'Murle', 'script': 'Latin'} 
    DID = {'name': 'Didinga', 'script': 'Latin'} 
    TEX = {'name': 'Tennet', 'script': 'Latin'} 
    VUT = {'name': 'Vute', 'script': 'Latin'} 
    TCC = {'name': 'Datooga', 'script': 'Latin'} 
    KNO = {'name': 'Kono', 'script': 'Latin'} 
    VAI = {'name': 'Vai', 'script': 'Vai'} 
    TUL = {'name': 'Kutule', 'script': 'Latin'} 
    BST = {'name': 'Basketo', 'script': 'Ethiopic'} 
    FFM = {'name': 'Fulfulde, Maasina', 'script': 'Latin'} 
    FUE = {'name': 'Fulfulde, Borgu', 'script': 'Latin'} 
    FUF = {'name': 'Pular', 'script': 'Latin'} 
    ZNE = {'name': 'Zande / paZande', 'script': 'Latin'} 
    GUW = {'name': 'Gun', 'script': 'Latin'} 
    AMH = {'name': 'Amharic', 'script': 'Ethiopic'} 
    BSP = {'name': 'Baga Sitemu', 'script': 'Latin'} 
    BZW = {'name': 'Basa', 'script': 'Latin'} 
    NHU = {'name': 'Noone', 'script': 'Latin'} 
    AVU = {'name': 'Avokaya', 'script': 'Latin'} 
    KBO = {'name': 'Keliko', 'script': 'Latin'} 
    LGG = {'name': 'Lugbara', 'script': 'Latin'} 
    LOG = {'name': 'Logo', 'script': 'Latin'} 
    LUC = {'name': 'Aringa', 'script': 'Latin'} 
    XNZ = {'name': 'Mattokki', 'script': 'Latin'} 
    UTH = {'name': 'u̱t-Hun', 'script': 'Latin'} 
    KYF = {'name': 'Kouya', 'script': 'Latin'} 
    IJN = {'name': 'Kalabari', 'script': 'Latin'} 
    OKR = {'name': 'Kirike', 'script': 'Latin'} 
    SHJ = {'name': 'Shatt', 'script': 'Latin'} 
    LRO = {'name': 'Laro', 'script': 'Latin'} 
    MKL = {'name': 'Mokole', 'script': 'Latin'} 
    YOR = {'name': 'Yoruba', 'script': 'Latin'} 
    BIN = {'name': 'Edo', 'script': 'Latin'} 
    ISH = {'name': 'Esan', 'script': 'Latin'} 
    ETU = {'name': 'Ejagham', 'script': 'Latin'} 
    FON = {'name': 'Fon', 'script': 'Latin'} 
    FUL = {'name': 'Fulah', 'script': 'Latin'} 
    GBR = {'name': 'Gbagyi', 'script': 'Latin'} 
    ATG = {'name': 'Ivbie North-Okpela-Arhe', 'script': 'Latin'} 
    KRW = {'name': 'Krahn, Western', 'script': 'Latin'} 
    WEC = {'name': 'Guéré', 'script': 'Latin'} 
    HAR = {'name': 'Harari', 'script': 'Latin'} 
    IGL = {'name': 'Igala', 'script': 'Latin'} 
    KTJ = {'name': 'Krumen, Plapo', 'script': 'Latin'} 
    TED = {'name': 'Krumen, Tepo', 'script': 'Latin'} 
    ASG = {'name': 'Cishingini', 'script': 'Latin'} 
    KDL = {'name': 'Tsikimba', 'script': 'Latin'} 
    TSW = {'name': 'Tsishingini', 'script': 'Latin'} 
    XRB = {'name': 'Karaboro, Eastern', 'script': 'Latin'} 
    KQS = {'name': 'Kisi', 'script': 'Latin'} 
    GBO = {'name': 'Grebo, Northern', 'script': 'Latin'} 
    LOM = {'name': 'Loma', 'script': 'Latin'} 
    ANV = {'name': 'Denya', 'script': 'Latin'} 
    KEN = {'name': 'Kenyang', 'script': 'Latin'} 
    MFI = {'name': 'Wandala', 'script': 'Latin'} 
    MEV = {'name': 'Maan / Mann', 'script': 'Latin'} 
    NGB = {'name': 'Ngbandi, Northern', 'script': 'Latin'} 
    FIA = {'name': 'Nobiin', 'script': 'Latin'} 
    NWB = {'name': 'Nyabwa', 'script': 'Latin'} 
    MDY = {'name': 'Maale', 'script': 'Ethiopic'} 
    EBR = {'name': 'Ebrié', 'script': 'Latin'} 
    SEF = {'name': 'Sénoufo, Cebaara', 'script': 'Latin'} 
    IRI = {'name': 'Rigwe', 'script': 'Latin'} 
    IZR = {'name': 'Izere', 'script': 'Latin'} 
    KCG = {'name': 'Tyap', 'script': 'Latin'} 
    SPP = {'name': 'Sénoufo, Supyire', 'script': 'Latin'} 
    MYK = {'name': 'Sénoufo, Mamara', 'script': 'Latin'} 
    DYI = {'name': 'Sénoufo, Djimini', 'script': 'Latin'} 
    SEV = {'name': 'Sénoufo, Nyarafolo', 'script': 'Latin'} 
    TGW = {'name': 'Sénoufo, Tagwana', 'script': 'Latin'} 
    TEM = {'name': 'Timne', 'script': 'Latin'} 
    TIV = {'name': 'Tiv', 'script': 'Latin'} 
    SGW = {'name': 'Sebat Bet Gurage', 'script': 'Latin'} 
    DNJ = {'name': 'Dan', 'script': 'Latin'} 
    WOL = {'name': 'Wolof', 'script': 'Latin'} 
    FAK = {'name': 'Fang', 'script': 'Latin'} 
    SOR = {'name': 'Somrai', 'script': 'Latin'} 
    BWR = {'name': 'Bura Pabir', 'script': 'Latin'} 
    KQP = {'name': 'Kimré', 'script': 'Latin'} 
    DAA = {'name': 'Dangaléat', 'script': 'Latin'} 
    MMY = {'name': 'Migaama', 'script': 'Latin'} 
    HBB = {'name': 'Nya huba', 'script': 'Latin'} 
    ABA = {'name': 'Abé / Abbey', 'script': 'Latin'} 
    ADJ = {'name': 'Adjukru  / Adioukrou', 'script': 'Latin'} 
    ATI = {'name': 'Attié', 'script': 'Latin'} 
    AVN = {'name': 'Avatime', 'script': 'Latin'} 
    NYB = {'name': 'Nyangbo', 'script': 'Latin'} 
    TCD = {'name': 'Tafi', 'script': 'Latin'} 
    BBA = {'name': 'Baatonum', 'script': 'Latin'} 
    BKY = {'name': 'Bokyi', 'script': 'Latin'} 
    BOM = {'name': 'Berom', 'script': 'Latin'} 
    ETX = {'name': 'Iten / Eten', 'script': 'Latin'} 
    GUD = {'name': 'Dida, Yocoboué', 'script': 'Latin'} 
    IGB = {'name': 'Ebira', 'script': 'Latin'} 
    ADA = {'name': 'Dangme', 'script': 'Latin'} 
    GAA = {'name': 'Ga', 'script': 'Latin'} 
    EWE = {'name': 'Éwé', 'script': 'Latin'} 
    GOL = {'name': 'Gola', 'script': 'Latin'} 
    YRE = {'name': 'Yaouré', 'script': 'Latin'} 
    IBO = {'name': 'Igbo', 'script': 'Latin'} 
    IKK = {'name': 'Ika', 'script': 'Latin'} 
    IKW = {'name': 'Ikwere', 'script': 'Latin'} 
    IQW = {'name': 'Ikwo', 'script': 'Latin'} 
    IZZ = {'name': 'Izii', 'script': 'Latin'} 
    KLU = {'name': 'Klao', 'script': 'Latin'} 
    GKP = {'name': 'Kpelle, Guinea', 'script': 'Latin'} 
    XPE = {'name': 'Kpelle', 'script': 'Latin'} 
    BOV = {'name': 'Tuwuli', 'script': 'Latin'} 
    AJG = {'name': 'Aja', 'script': 'Latin'} 
    KRS = {'name': 'Gbaya', 'script': 'Latin'} 
    XED = {'name': 'Hdi', 'script': 'Latin'} 
    NIY = {'name': 'Ngiti', 'script': 'Latin'} 
    AFR = {'name': 'Afrikaans', 'script': 'Latin'} 
    KNF = {'name': 'Mankanya', 'script': 'Latin'} 
    MOY = {'name': 'Shekkacho', 'script': 'Latin'} 
    ISO = {'name': 'Isoko', 'script': 'Latin'} 
    OKE = {'name': 'Okpe', 'script': 'Latin'} 
    URH = {'name': 'Urhobo', 'script': 'Latin'} 
    SUS = {'name': 'Sosoxui', 'script': 'Latin'} 
    YAL = {'name': 'Yalunka', 'script': 'Latin'} 
    BSC = {'name': 'Oniyan', 'script': 'Latin'} 
    COU = {'name': 'Wamey', 'script': 'Latin'}
    WIB = {'name': 'Toussian, Southern', 'script': 'Latin'}
    MOA = {'name': 'Mwan', 'script': 'Latin'}
    TTR = {'name': 'Nyimatli', 'script': 'Latin'}
    KUB = {'name': 'Kutep', 'script': 'Latin'}
    HAU = {'name': 'Hausa', 'script': 'Latin'}
    BCW = {'name': 'Bana', 'script': 'Latin'}
    KVJ = {'name': 'Psikye', 'script': 'Latin'}
    GIZ = {'name': 'South Giziga', 'script': 'Latin'}
    GND = {'name': 'Zulgo-gemzek', 'script': 'Latin'}
    MAF = {'name': 'Mafa', 'script': 'Latin'}
    MEQ = {'name': 'Merey', 'script': 'Latin'}
    MFH = {'name': 'Matal', 'script': 'Latin'}
    MFK = {'name': 'Mofu, North', 'script': 'Latin'}
    MIF = {'name': 'Mofu-Gudur', 'script': 'Latin'}
    MLR = {'name': 'Vame', 'script': 'Latin'}
    MQB = {'name': 'Mbuko', 'script': 'Latin'}
    MUY = {'name': 'Muyang', 'script': 'Latin'}
    HNA = {'name': 'Mina', 'script': 'Latin'}
    BCY = {'name': 'Bacama', 'script': 'Latin'}
    GDE = {'name': 'Gude', 'script': 'Latin'}
    MOZ = {'name': 'Mukulu', 'script': 'Latin'}
    BIB = {'name': 'Bisa', 'script': 'Latin'}
    BQC = {'name': 'Boko', 'script': 'Latin'}
    BUS = {'name': 'Bokobaru', 'script': 'Latin'}
    NDV = {'name': 'Ndut', 'script': 'Latin'}
    SNF = {'name': 'Noon', 'script': 'Latin'}
    BYF = {'name': 'Bété', 'script': 'Latin'}
    LIA = {'name': 'Limba, West-Central', 'script': 'Latin'}
    MLG = {'name': 'Malagasy', 'script': 'Latin'}
    TIR = {'name': 'Tigrinya', 'script': 'Ethiopic'}
    RIF = {'name': 'Tarifit', 'script': 'Arabic'}
    SBD = {'name': 'Samo, Southern', 'script': 'Latin'}
    NHR = {'name': 'Naro', 'script': 'Latin'}
    LMD = {'name': 'Lumun', 'script': 'Latin'}
    SHI = {'name': 'Tachelhit', 'script': 'Latin'}
    GID = {'name': 'Gidar', 'script': 'Latin'}
    XAN = {'name': 'Xamtanga', 'script': 'Ethiopic'}
    HGM = {'name': 'Hai|ǁom', 'script': 'Latin'}
    SID = {'name': 'Sidama', 'script': 'Latin'}
    KAB = {'name': 'Kabyle', 'script': 'Latin'}
    XTC = {'name': 'Katcha-Kadugli-Miri', 'script': 'Latin'}
    KBY = {'name': 'Kanuri, Manga', 'script': 'Latin'}
    KRI = {'name': 'Krio', 'script': 'Latin'}
    WES = {'name': 'Pidgin, Cameroon', 'script': 'Latin'}
    PCM = {'name': 'Nigerian Pidgin', 'script': 'Latin'}
    NAQ = {'name': 'Khoekhoe', 'script': 'Latin'}
    THV = {'name': 'Tamahaq, Tahaggart', 'script': 'Latin'}
    GAX = {'name': 'Oromo, Borana-Arsi-Guji', 'script': 'Ethiopic / Latin'}
    GAZ = {'name': 'Oromo, West Central', 'script': 'Ethiopic / Latin'}
    ORM = {'name': 'Oromo', 'script': 'Ethiopic / Latin'}
    REL = {'name': 'Rendille', 'script': 'Latin'}
    AAR = {'name': 'Afar / Qafar', 'script': 'Latin'}
    SOM = {'name': 'Somali', 'script': 'Latin'}
    TAQ = {'name': 'Tamasheq', 'script': 'Latin'}
    TTQ = {'name': 'Tawallammat', 'script': 'Latin'}
    DSH = {'name': 'Daasanach', 'script': 'Latin'}
    MCN = {'name': 'Masana / Massana', 'script': 'Latin'}
    MPG = {'name': 'Marba', 'script': 'Latin'}
    BDS = {'name': 'Burunge', 'script': 'Latin'}
    SES = {'name': 'Songhay, Koyraboro Senni', 'script': 'Latin'}
    COP = {'name': 'Coptic', 'script': 'Coptic'}
    BER = {'name': 'Berber', 'script': 'Latin'}
    CRS = {'name': 'Seychelles Creole', 'script': 'Latin'}
    MFE = {'name': 'Morisyen / Mauritian Creole', 'script': 'Latin'}
    KTU = {'name': 'Kikongo', 'script': 'Latin'}
    SAG = {'name': 'Sango', 'script': 'Latin'}
    KEA = {'name': 'Kabuverdianu', 'script': 'Latin'}
    POV = {'name': 'Guinea-Bissau Creole', 'script': 'Latin'}


class LanguageInfo:
    _info: Final[Enum] = Language
    Info = TypedDict('Info', {'name': str, 'script': str})
    def __init__(self, label_file: str):
        with open(label_file, "r") as f:
            self.idx2code = {
                idx+4: code.split()[0]
                for idx, code in enumerate(f.readlines())
            }
    
    def get_language_info_from_idx(self, idx_or_code: int | str) -> Info:
        laguage_code = idx_or_code.upper() if isinstance(idx_or_code, str) \
            else self.idx2code[idx_or_code].upper()
        return self._info[laguage_code].value
    
    def __getitem__(self, idx_or_code: int | str) -> Info:
        return self.get_language_info_from_idx(idx_or_code)
