

module Normalization where



import Control.Applicative ((<$>),(<*>))
import Data.Foldable (sum)
import Prelude hiding (sum)

-- -- | see http://www.ccl.net/cca/documents/basis-sets/basis.html 
-- normGlobal :: CGF -> CGF
-- normGlobal cgf@(CGF ps _) =  newCGF { getPrimitives = newPrimitives}
--   where newPrimitives = zip newCs es 
--         newCs         = map (/normaG) cs
--         (cs,es)       = unzip . getPrimitives $ newCGF 
--         newCGF        = normCoeff cgf   
--         r0            = [0,0,0] 
--         normaG        = sqrt . recip $  sijContracted r0 r0 cgf cgf 


{- |The norm of each contracted is given by the following equation
    N = sqrt $ ((2l -1)!! (2m-1)!! (2n-1)!!)/(4*expo)^(l+m+n)  * (pi/(2*e))**1.5
    where expo is the exponential factor of the contracted -}
normCoeff :: [Double] -> [Double] -> [Double]
normCoeff es cs = zipWith fun es cs 
  where fun = \e c -> c * (2*e/pi)**0.75


sij :: [Double] -> [Double] -> Double
sij es cs = sqrt . recip . sum $ fun <$> z1 <*> z1
  where fun (e1,c1) (e2,c2) = c1*c2 * (4*e1*e2/pi^2)**0.75  * (pi/(e1+e2))**1.5
        z1 = zip es cs

sij2 :: [Double] -> [Double] -> Double
sij2 es cs = sqrt . recip . sum $ fun <$> z1 <*> z1
  where fun (e1,c1) (e2,c2) = c1*c2 *  (pi/(e1+e2))**1.5
        z1 = zip es cs


normalization :: [Double] -> [Double] -> [Double]
normalization es cs  = map (*n) cs
 where n = sij es cs 

normalization2 :: [Double] -> [Double] -> [Double]
normalization2 es cs  = map (*n) $ cs
 where n  = sij2 es xs 
       xs = normCoeff es cs

normalization3 :: [Double] -> [Double] -> [Double]
normalization3 es cs  = map (*n) $ xs
 where n  = sij2 es xs 
       xs = normCoeff es cs


test :: [Double] -> [Double] -> Double
test es cs = sum $ fun <$> z1 <*> z1
    where xs = normalization3 es cs
          fun (e1,c1) (e2,c2) = c1*c2 *  (pi/(e1+e2))**1.5
          z1 = zip es xs

test2 :: [Double] -> [Double] -> Double
test2 es cs = sum $ fun <$> z1 <*> z1
    where xs = normCoeff es cs
          fun (e1,c1) (e2,c2) = c1*c2 *  (pi/(e1+e2))**1.5
          z1 = zip es xs

facOdd ::Int -> Double
facOdd  i | i `rem`2 == 0  = error "Factorial Odd function required an odd integer as input"
          | otherwise  = case compare i 2 of
                             LT -> 1
                             GT-> let k = (1 + i) `div ` 2
                                  in (fromIntegral $ fac (2*k)) /  (2.0^k * (fromIntegral $ fac k))

-- | Factorial function
fac :: Int -> Int
fac i | i < 0   = error "The factorial function is defined only for natural numbers"
      | i == 0  = 1
      | otherwise = product [1..i]

es = [ 13575.349682, 2035.2333680, 463.22562359, 131.20019598,  42.853015891, 15.584185766]

cs = [0.22245814352E-03, 0.17232738252E-02, 0.89255715314E-02, 0.35727984502E-01, 0.11076259931,0.24295627626]

csN = [0.5435906373416398,1.0145544095767811,1.7315732031397297,2.6910376292325497,3.6044547130790408,3.70255569805205]

-- csC_N = [0.19939839661626194, 0.37215600964154844, 0.63517081144965803, 0.98711892726692829, 1.3221759789777903, 1.3581611602460502]


-- cs2 = [6.064550360359116e-4,4.697908888498168e-3,2.4332477647551906e-2,9.739996830775899e-2,0.30195584253780383,0.6623360913787472]

-- esO = [ 27032.3828125 ,   4052.38720703,    922.32720947,    261.24072266,      85.35464478,     31.03503609]
-- csC =[  0.21726302465E-03 ,0.16838662199E-02  ,0.87395616265E-02  ,0.35239968808E-01  ,0.11153519115   ,0.25588953961]
